#!/usr/bin/env python3
"""Single-Dynamixel / single-SparkFun-IMU mapper.

The hardware layer is intentionally independent from the GLFW/ImGui layer so
that the parser, serial protocol and CSV/XLSX format can be tested without a
display or connected hardware.  GLFW/ImGui is preferred, with a native
Tkinter UI as a fallback.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # The module remains importable for parser-only tests.
    serial = None  # type: ignore[assignment]
    list_ports = None  # type: ignore[assignment]

try:
    import dynamixel_sdk as dynamixel
except ImportError:  # Hardware is optional until the user presses Connect.
    dynamixel = None  # type: ignore[assignment]

try:
    import openpyxl
except ImportError:  # XLSX export reports a useful error until installed.
    openpyxl = None  # type: ignore[assignment]

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ImportError:  # Tk is optional when running on a minimal/headless host.
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

try:
    # PyOpenGL's context tracker needs GLX to match the X11 context created by
    # GLFW.  Without this on WSL/Linux, GlfwRenderer can see a window but
    # reports "no valid context" during glVertexAttribPointer().
    if os.environ.get("DISPLAY"):
        os.environ["PYOPENGL_PLATFORM"] = "glx"
    import glfw
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
except Exception as error:  # Importing map.py must still work headlessly.
    glfw = None  # type: ignore[assignment]
    imgui = None  # type: ignore[assignment]
    GlfwRenderer = None  # type: ignore[assignment]
    GUI_IMPORT_ERROR = error
else:
    GUI_IMPORT_ERROR = None


# ------------------------------- Constants ------------------------------

PROTOCOL_VERSION = 2.0
DEFAULT_BAUDRATE = 57600
DEFAULT_IMU_BAUDRATE = 115200
DEFAULT_MOTOR_ID = 3
DEFAULT_POSITION_DEG = 180.0
DEFAULT_PROFILE_VELOCITY = 50
POLL_PERIOD_SEC = 0.10
REACH_TOLERANCE_DEG = 2.0
SCENARIO_TIMEOUT_SEC = 20.0

# X-series and XL430-compatible addresses.  Present Load (126) is a signed
# 16-bit value on X-series and is also the commonly used load/current field on
# XL430-class actuators.  Change these constants for another control table.
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_LOAD = 126
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_FILE = ROOT_DIR / "script.txt"

CSV_COLUMNS = [
    "timestamp_iso",
    "unix_time",
    "motor_position_deg",
    "motor_load",
    "imu_roll_deg",
    "imu_pitch_deg",
    "imu_yaw_deg",
    "imu_qw",
    "imu_qx",
    "imu_qy",
    "imu_qz",
    "imu_magnetometer",
]


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def clamp_position(value: float) -> float:
    return max(0.0, min(360.0, float(value)))


def degree_to_tick(value: float) -> int:
    return int(round(clamp_position(value) * 4095.0 / 360.0))


def tick_to_degree(value: int) -> float:
    return round((int(value) & 0xFFFFFFFF) * 360.0 / 4095.0, 2)


def signed16(value: int) -> int:
    value = int(value) & 0xFFFF
    return value - 65536 if value >= 32768 else value


def quaternion_to_euler(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in degrees using the same convention as imu.ino."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.degrees(math.asin(sinp))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


# ----------------------------- IMU serial layer --------------------------

@dataclass
class ImuReading:
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    magnetometer: Optional[bool] = None
    updated_at: float = 0.0

    @property
    def age(self) -> float:
        if self.updated_at <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - self.updated_at)

    def copy(self) -> "ImuReading":
        return ImuReading(
            self.qw, self.qx, self.qy, self.qz,
            self.roll, self.pitch, self.yaw,
            self.magnetometer, self.updated_at,
        )


IMU_QUATERNION_RE = re.compile(
    r"IMU(?P<index>[12])\s*>\s*Q\(\s*"
    r"(?P<qw>[-+]?\d*\.?\d+)\s*,\s*"
    r"(?P<qx>[-+]?\d*\.?\d+)\s*,\s*"
    r"(?P<qy>[-+]?\d*\.?\d+)\s*,\s*"
    r"(?P<qz>[-+]?\d*\.?\d+)\s*\)",
    re.IGNORECASE,
)


class ImuSerialReader:
    """Reader for the CSV lines emitted by new/imu.ino."""

    def __init__(self, on_log: Callable[[str], None] = print) -> None:
        self.on_log = on_log
        self.port_name = ""
        self.baudrate = DEFAULT_IMU_BAUDRATE
        self._serial = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._reading = ImuReading()
        self._last_parse_error = ""

    @property
    def connected(self) -> bool:
        return self._serial is not None and self._thread is not None

    def connect(self, port_name: str, baudrate: int = DEFAULT_IMU_BAUDRATE) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed; install pyserial first")
        self.disconnect()
        port_name = str(port_name).strip()
        if not port_name:
            raise ValueError("IMU port is empty")
        self._serial = serial.Serial(port_name, int(baudrate), timeout=0.2)
        self.port_name = port_name
        self.baudrate = int(baudrate)
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="imu-reader", daemon=True)
        self._thread.start()
        self.on_log(f"IMU connected: {port_name} @ {baudrate}")
        self.send_command("STATUS")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None

    def send_command(self, command: str) -> None:
        if self._serial is None:
            raise RuntimeError("IMU is not connected")
        payload = (str(command).strip() + "\n").encode("ascii")
        with self._lock:
            self._serial.write(payload)

    def set_magnetometer(self, enabled: bool) -> None:
        self.send_command("MAG ON" if enabled else "MAG OFF")

    def set_power(self, enabled: bool) -> None:
        self.send_command("IMU ON" if enabled else "IMU OFF")

    def snapshot(self) -> ImuReading:
        with self._lock:
            return self._reading.copy()

    @staticmethod
    def parse_line(line: str) -> Optional[tuple[int, ImuReading]]:
        """Parse both the new CSV protocol and imu_mapper's old Q(...) line."""
        text = str(line).strip()
        if not text:
            return None
        fields = [field.strip() for field in text.split(",")]
        if len(fields) >= 10 and re.fullmatch(r"IMU[12]", fields[0], re.IGNORECASE):
            index = int(fields[0][-1])
            try:
                qw, qx, qy, qz = (float(fields[i]) for i in range(2, 6))
                roll, pitch, yaw = (float(fields[i]) for i in range(6, 9))
                mag = fields[9].upper() in {"1", "ON", "TRUE", "YES"}
            except (TypeError, ValueError):
                return None
            return index, ImuReading(
                qw, qx, qy, qz, roll, pitch, yaw, mag, time.monotonic()
            )

        # Compatibility with aaa.ino/bbb.py's legacy Game Rotation Vector
        # stream: I,ax,ay,az,gx,gy,gz,qi,qj,qk,qr,status
        if len(fields) >= 11 and fields[0].upper() == "I":
            try:
                qx, qy, qz, qw = (float(fields[i]) for i in range(7, 11))
            except (TypeError, ValueError):
                return None
            roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
            return 1, ImuReading(
                qw, qx, qy, qz, roll, pitch, yaw, False, time.monotonic()
            )

        match = IMU_QUATERNION_RE.search(text)
        if match is None:
            return None
        qw = float(match.group("qw"))
        qx = float(match.group("qx"))
        qy = float(match.group("qy"))
        qz = float(match.group("qz"))
        roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
        return int(match.group("index")), ImuReading(
            qw, qx, qy, qz, roll, pitch, yaw, None, time.monotonic()
        )

    def handle_line(self, line: str) -> None:
        parsed = self.parse_line(line)
        if parsed is None:
            if line.strip() and not line.lstrip().upper().startswith(("STATUS", "INIT", "DATA_FORMAT", "MAGNETOMETER")):
                self._last_parse_error = line.strip()
            return
        index, reading = parsed
        with self._lock:
            # The single sensor is exposed as IMU1 by new/imu.ino.  Accepting
            # IMU2 here keeps the reader tolerant of an older dual-sensor
            # firmware during a transition, while still storing one reading.
            self._reading = reading

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._serial is None:
                    return
                raw = self._serial.readline()
                if not raw:
                    continue
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    self.handle_line(text)
                    if text.startswith(("STATUS", "MAGNETOMETER", "IMU_POWER", "ERROR")):
                        self.on_log(f"IMU: {text}")
            except Exception as error:
                self.on_log(f"IMU read error: {error}")
                time.sleep(0.25)


# ----------------------------- Dynamixel layer ---------------------------

@dataclass
class MotorReading:
    position_deg: Optional[float] = None
    load: Optional[int] = None
    error: str = ""


class DynamixelMotorController:
    """Protocol 2.0 controller for the one Dynamixel used by this project."""

    def __init__(
        self,
        motor_id: int = DEFAULT_MOTOR_ID,
        baudrate: int = DEFAULT_BAUDRATE,
        on_log: Callable[[str], None] = print,
    ) -> None:
        motor_id = int(motor_id)
        if motor_id != DEFAULT_MOTOR_ID:
            raise ValueError(f"this project supports exactly one Dynamixel: ID {DEFAULT_MOTOR_ID}")
        self.motor_id = motor_id
        self.baudrate = int(baudrate)
        self.on_log = on_log
        self.port_name = ""
        self._port = None
        self._packet = None
        self._lock = threading.RLock()
        self.torque_enabled = False
        self.profile_velocity = 50

    @property
    def connected(self) -> bool:
        return self._port is not None and self._packet is not None

    def connect(self, port_name: str) -> None:
        if dynamixel is None:
            raise RuntimeError("dynamixel-sdk is not installed; install it first")
        self.disconnect()
        port_name = str(port_name).strip()
        if not port_name:
            raise ValueError("motor port is empty")
        port = dynamixel.PortHandler(port_name)
        packet = dynamixel.PacketHandler(PROTOCOL_VERSION)
        if not port.openPort():
            raise RuntimeError(f"cannot open motor port: {port_name}")
        if not port.setBaudRate(self.baudrate):
            port.closePort()
            raise RuntimeError(f"cannot set motor baudrate: {self.baudrate}")
        with self._lock:
            self._port = port
            self._packet = packet
            self.port_name = port_name
            self.torque_enabled = False
        self.on_log(f"Motor connected: {port_name}, ID={self.motor_id}")

    def disconnect(self) -> None:
        with self._lock:
            if self._port is not None:
                try:
                    self._write_torque_locked(False)
                except Exception:
                    pass
                try:
                    self._port.closePort()
                except Exception:
                    pass
            self._port = None
            self._packet = None
            self.torque_enabled = False

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("motor is not connected")

    def _write_torque_locked(self, enabled: bool) -> bool:
        if self._port is None or self._packet is None:
            return False
        success = True
        try:
            result, error = self._packet.write1ByteTxRx(
                self._port, self.motor_id, ADDR_TORQUE_ENABLE,
                TORQUE_ENABLE if enabled else TORQUE_DISABLE,
            )
            success = result == dynamixel.COMM_SUCCESS and error == 0
        except Exception as exc:
            success = False
            self.on_log(f"Torque ID {self.motor_id} error: {exc}")
        self.torque_enabled = bool(enabled and success)
        if success:
            self.on_log(f"Motor torque {'ON' if enabled else 'OFF'}")
        return success

    def set_torque(self, enabled: bool) -> bool:
        with self._lock:
            self._require_connected()
            return self._write_torque_locked(bool(enabled))

    def set_profile_velocity(self, value: int) -> None:
        velocity = max(0, min(32767, int(value)))
        with self._lock:
            self._require_connected()
            if self._port is None or self._packet is None:
                return
            result, error = self._packet.write4ByteTxRx(
                self._port, self.motor_id, ADDR_PROFILE_VELOCITY, velocity
            )
            if result != dynamixel.COMM_SUCCESS or error != 0:
                raise RuntimeError(f"cannot set velocity for motor {self.motor_id}")
            self.profile_velocity = velocity

    def set_position(self, position: float) -> None:
        with self._lock:
            self._require_connected()
            if self._port is None or self._packet is None:
                return
            result, error = self._packet.write4ByteTxRx(
                self._port, self.motor_id, ADDR_GOAL_POSITION, degree_to_tick(position)
            )
            if result != dynamixel.COMM_SUCCESS or error != 0:
                raise RuntimeError(f"cannot move motor {self.motor_id}")

    def set_positions(self, positions: Iterable[float] | float) -> None:
        """Compatibility wrapper accepting one scalar or a one-item iterable."""
        if isinstance(positions, (int, float)):
            self.set_position(float(positions))
            return
        values = tuple(positions)
        if len(values) != 1:
            raise ValueError("exactly one motor position is required")
        self.set_position(float(values[0]))

    def read_state(self) -> MotorReading:
        reading = MotorReading()
        with self._lock:
            if not self.connected or self._port is None or self._packet is None:
                return reading
            try:
                position, result, error = self._packet.read4ByteTxRx(
                    self._port, self.motor_id, ADDR_PRESENT_POSITION
                )
                if result != dynamixel.COMM_SUCCESS or error != 0:
                    reading.error = f"position read failed (id={self.motor_id})"
                    return reading
                load, result, error = self._packet.read2ByteTxRx(
                    self._port, self.motor_id, ADDR_PRESENT_LOAD
                )
                reading.position_deg = tick_to_degree(position)
                if result == dynamixel.COMM_SUCCESS and error == 0:
                    reading.load = signed16(load)
                else:
                    reading.error = f"load read failed (id={self.motor_id})"
            except Exception as exc:
                reading.error = str(exc)
        return reading


# ------------------------------- Scenario DSL ----------------------------

SCENARIO_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class MotionCommand:
    name: str
    args: tuple
    line: int
    source: str


def _number(value: str, label: str, minimum: float, maximum: float, source: str, line: int) -> float:
    try:
        number = float(value.strip().rstrip("sS"))
    except ValueError as error:
        raise ScenarioError(f"{source}:{line}: {label} is not a number: {value!r}") from error
    if not minimum <= number <= maximum:
        raise ScenarioError(f"{source}:{line}: {label} must be in {minimum}..{maximum}")
    return number


def _split_values(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _pose(values: list[str], source: str, line: int) -> float:
    if len(values) == 1:
        return _number(values[0], "angle", 0.0, 360.0, source, line)
    raise ScenarioError(f"{source}:{line}: move requires exactly one angle")


def read_scenario_sections(path: Path) -> dict[str, list[tuple[int, str]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ScenarioError(f"cannot read scenario file {path}: {error}") from error
    sections: dict[str, list[tuple[int, str]]] = {}
    current: Optional[str] = None
    for line_number, original in enumerate(lines, start=1):
        stripped = original.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip()
            if not SCENARIO_NAME_RE.fullmatch(name):
                raise ScenarioError(f"{path}:{line_number}: invalid scenario name")
            if name in sections:
                raise ScenarioError(f"{path}:{line_number}: duplicate scenario {name}")
            sections[name] = []
            current = name
        elif stripped and not stripped.startswith(("#", "//")):
            if current is None:
                raise ScenarioError(f"{path}:{line_number}: command before a section")
            sections[current].append((line_number, original))
    return sections


def load_scenario(path: Path, name: str) -> list[MotionCommand]:
    if not SCENARIO_NAME_RE.fullmatch(name):
        raise ScenarioError(f"unsafe scenario name: {name!r}")
    sections = read_scenario_sections(path)
    if name not in sections:
        raise ScenarioError(f"unknown scenario: {name}")
    commands: list[MotionCommand] = []
    source = str(path)
    for line_number, original in sections[name]:
        content = original.split("//", 1)[0].split("#", 1)[0].strip()
        if not content:
            continue

        match = re.fullmatch(r"/?(?:move|pose)\s*\((.*)\)", content, re.IGNORECASE)
        if match:
            commands.append(MotionCommand("move", (_pose(_split_values(match.group(1)), source, line_number),), line_number, source))
            continue
        match = re.fullmatch(r"/?cycle\s*\((.*)\)", content, re.IGNORECASE)
        if match:
            values = _split_values(match.group(1))
            if len(values) < 3:
                raise ScenarioError(f"{source}:{line_number}: cycle requires a route and pause")
            route = tuple(_number(value, "cycle route angle", 0.0, 360.0, source, line_number) for value in values[:-1])
            pause = _number(values[-1], "cycle pause", 0.0, 3600.0, source, line_number)
            commands.append(MotionCommand("cycle", (route, pause), line_number, source))
            continue
        match = re.fullmatch(r"/?seqmove\s*\((.*)\)", content, re.IGNORECASE)
        if match:
            values = _split_values(match.group(1))
            if len(values) != 3:
                raise ScenarioError(f"{source}:{line_number}: seqmove requires start,end,pause")
            start = _number(values[0], "seqmove start", 0.0, 360.0, source, line_number)
            end = _number(values[1], "seqmove end", 0.0, 360.0, source, line_number)
            pause = _number(values[2], "seqmove pause", 0.0, 3600.0, source, line_number)
            commands.append(MotionCommand("seqmove", (start, end, pause), line_number, source))
            continue

        tokens = content.replace(",", " ").split()
        command_name = tokens[0].lstrip("/").lower()
        values = tokens[1:]
        if command_name in {"move", "pose"}:
            commands.append(MotionCommand("move", (_pose(values, source, line_number),), line_number, source))
        elif command_name == "pause" and len(values) == 1:
            commands.append(MotionCommand("pause", (_number(values[0], "pause", 0.0, 3600.0, source, line_number),), line_number, source))
        elif command_name == "velocity" and len(values) == 1:
            commands.append(MotionCommand("velocity", (int(_number(values[0], "velocity", 0.0, 32767.0, source, line_number)),), line_number, source))
        elif command_name == "run" and len(values) == 1 and SCENARIO_NAME_RE.fullmatch(values[0]):
            commands.append(MotionCommand("run", (values[0],), line_number, source))
        elif command_name == "mark" and len(values) == 1 and SCENARIO_NAME_RE.fullmatch(values[0]):
            commands.append(MotionCommand("mark", (values[0],), line_number, source))
        else:
            raise ScenarioError(f"{source}:{line_number}: unsupported command: {content}")
    if not commands:
        raise ScenarioError(f"scenario is empty: {name}")
    return commands


def expand_scenario(path: Path, name: str, stack: tuple[str, ...] = ()) -> list[MotionCommand]:
    if name in stack:
        raise ScenarioError(f"recursive run detected: {' -> '.join((*stack, name))}")
    expanded: list[MotionCommand] = []
    for command in load_scenario(path, name):
        if command.name == "run":
            expanded.extend(expand_scenario(path, command.args[0], (*stack, name)))
        else:
            expanded.append(command)
    return expanded


class ScenarioRunner:
    def __init__(self, motor: DynamixelMotorController, on_log: Callable[[str], None]) -> None:
        self.motor = motor
        self.on_log = on_log
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.status = "idle"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, path: Path, name: str) -> None:
        if self.running:
            raise RuntimeError("a scenario is already running")
        commands = expand_scenario(path, name)
        self._stop.clear()
        self.status = f"running {name}"
        self._thread = threading.Thread(
            target=self._run, args=(name, commands), name="scenario-runner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.status = "stopped"

    def _wait(self, seconds: float) -> bool:
        return not self._stop.wait(max(0.0, float(seconds)))

    def _move_and_wait(self, target: float) -> bool:
        self.motor.set_position(target)
        started = time.monotonic()
        while not self._stop.is_set() and time.monotonic() - started < SCENARIO_TIMEOUT_SEC:
            reading = self.motor.read_state()
            if reading.position_deg is not None:
                if abs(float(reading.position_deg) - target) <= REACH_TOLERANCE_DEG:
                    return True
            if not self._wait(POLL_PERIOD_SEC):
                return False
        return self._stop.is_set() or False

    def _run(self, name: str, commands: list[MotionCommand]) -> None:
        last_target: Optional[float] = None
        try:
            for command in commands:
                if self._stop.is_set():
                    break
                if command.name == "move":
                    target = command.args[0]
                    self.on_log(f"AUTO move M={target:.1f}")
                    if not self._move_and_wait(target):
                        if self._stop.is_set():
                            break
                        raise RuntimeError(f"target not reached: {target}")
                    last_target = target
                elif command.name == "pause":
                    if not self._wait(command.args[0]):
                        break
                elif command.name == "velocity":
                    self.motor.set_profile_velocity(command.args[0])
                    self.on_log(f"AUTO velocity={command.args[0]}")
                elif command.name == "mark":
                    self.on_log(f"AUTO mark={command.args[0]}")
                elif command.name == "cycle":
                    route, pause = command.args
                    targets = list(route) + list(reversed(route[:-1]))
                    for angle in targets:
                        target = angle
                        self.on_log(f"AUTO cycle M={angle:.1f}")
                        if not self._move_and_wait(target):
                            if self._stop.is_set():
                                break
                            raise RuntimeError(f"cycle target not reached: {target}")
                        last_target = target
                        if not self._wait(pause):
                            break
                elif command.name == "seqmove":
                    start, end, pause = command.args
                    targets = [start, end] if last_target != start else [end]
                    for target in targets:
                        self.on_log(f"AUTO seqmove M={target:.1f}")
                        if not self._move_and_wait(target):
                            if self._stop.is_set():
                                break
                            raise RuntimeError(f"seqmove target not reached: {target}")
                        last_target = target
                        if not self._wait(pause):
                            break
            self.status = "stopped" if self._stop.is_set() else "finished"
            self.on_log(f"AUTO {self.status}: {name}")
        except Exception as error:
            self.status = f"error: {error}"
            self.on_log(f"AUTO error: {error}")


# ------------------------------ CSV recorder -----------------------------

class CsvRecorder:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.recording = False
        self.rows: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self.rows.clear()
            self.recording = True

    def finish(self) -> int:
        with self._lock:
            self.recording = False
            return len(self.rows)

    def append(self, row: dict[str, object]) -> None:
        with self._lock:
            if self.recording:
                self.rows.append(dict(row))

    def export(self, path: Optional[str] = None) -> tuple[str, int]:
        target = Path(path or self.path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self.rows)
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        self.path = str(target)
        return str(target), len(rows)

    def export_xlsx(self, path: Optional[str] = None) -> tuple[str, int]:
        if openpyxl is None:
            raise RuntimeError("openpyxl is not installed; run: python3 -m pip install openpyxl")
        default_path = Path(self.path).with_suffix(".xlsx")
        target = Path(path or default_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self.rows)
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "imu_motor"
        sheet.append(CSV_COLUMNS)
        for row in rows:
            sheet.append([row.get(column, "") for column in CSV_COLUMNS])
        workbook.save(str(target))
        return str(target), len(rows)


# ------------------------------ GUI helpers ------------------------------

def _imgui_bool(result: object) -> bool:
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


def _imgui_open(result: object) -> bool:
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


def _set_imgui_color(style: object, name: str, rgba: tuple[float, float, float, float]) -> None:
    if imgui is None:
        return
    color_id = getattr(imgui, name, None)
    if color_id is None:
        return
    try:
        style.colors[color_id] = rgba
    except Exception:
        pass


def _install_imgui_theme() -> None:
    """Apply the light Elesim palette and compact panel geometry."""
    if imgui is None:
        return
    style = imgui.get_style()
    for attr, value in (
        ("window_rounding", 6.0),
        ("child_rounding", 5.0),
        ("frame_rounding", 4.0),
        ("grab_rounding", 4.0),
        ("popup_rounding", 5.0),
        ("scrollbar_rounding", 6.0),
        ("tab_rounding", 5.0),
        ("window_border_size", 1.0),
        ("child_border_size", 1.0),
        ("frame_border_size", 1.0),
    ):
        if hasattr(style, attr):
            setattr(style, attr, value)
    for attr, value in (
        ("item_spacing", (8.0, 10.5)),
        ("frame_padding", (8.0, 6.0)),
        ("window_padding", (11.0, 13.5)),
        ("cell_padding", (7.0, 6.0)),
    ):
        if hasattr(style, attr):
            setattr(style, attr, value)

    colors = {
        "COLOR_TEXT": (0.10, 0.11, 0.13, 1.00),
        "COLOR_TEXT_DISABLED": (0.48, 0.50, 0.54, 1.00),
        "COLOR_WINDOW_BACKGROUND": (0.94, 0.95, 0.96, 1.00),
        "COLOR_CHILD_BACKGROUND": (0.985, 0.985, 0.99, 1.00),
        "COLOR_POPUP_BACKGROUND": (1.00, 1.00, 1.00, 0.98),
        "COLOR_BORDER": (0.74, 0.76, 0.80, 1.00),
        "COLOR_BORDER_SHADOW": (1.00, 1.00, 1.00, 0.00),
        "COLOR_FRAME_BACKGROUND": (1.00, 1.00, 1.00, 1.00),
        "COLOR_FRAME_BACKGROUND_HOVERED": (0.91, 0.95, 1.00, 1.00),
        "COLOR_FRAME_BACKGROUND_ACTIVE": (0.84, 0.90, 1.00, 1.00),
        "COLOR_TITLE_BACKGROUND": (0.88, 0.89, 0.91, 1.00),
        "COLOR_TITLE_BACKGROUND_ACTIVE": (0.82, 0.86, 0.92, 1.00),
        "COLOR_TITLE_BACKGROUND_COLLAPSED": (0.90, 0.91, 0.93, 1.00),
        "COLOR_MENU_BAR_BACKGROUND": (0.91, 0.92, 0.94, 1.00),
        "COLOR_SCROLLBAR_BACKGROUND": (0.93, 0.94, 0.95, 1.00),
        "COLOR_SCROLLBAR_GRAB": (0.70, 0.72, 0.76, 1.00),
        "COLOR_SCROLLBAR_GRAB_HOVERED": (0.62, 0.65, 0.70, 1.00),
        "COLOR_SCROLLBAR_GRAB_ACTIVE": (0.52, 0.56, 0.62, 1.00),
        "COLOR_CHECK_MARK": (0.00, 0.45, 0.95, 1.00),
        "COLOR_SLIDER_GRAB": (0.00, 0.48, 1.00, 1.00),
        "COLOR_SLIDER_GRAB_ACTIVE": (0.00, 0.36, 0.86, 1.00),
        "COLOR_BUTTON": (0.90, 0.91, 0.93, 1.00),
        "COLOR_BUTTON_HOVERED": (0.82, 0.89, 0.98, 1.00),
        "COLOR_BUTTON_ACTIVE": (0.70, 0.82, 0.98, 1.00),
        "COLOR_HEADER": (0.86, 0.88, 0.91, 1.00),
        "COLOR_HEADER_HOVERED": (0.78, 0.86, 0.98, 1.00),
        "COLOR_HEADER_ACTIVE": (0.66, 0.78, 0.96, 1.00),
        "COLOR_SEPARATOR": (0.78, 0.80, 0.84, 1.00),
        "COLOR_SEPARATOR_HOVERED": (0.52, 0.66, 0.86, 1.00),
        "COLOR_SEPARATOR_ACTIVE": (0.34, 0.54, 0.82, 1.00),
        "COLOR_TAB": (0.86, 0.88, 0.91, 1.00),
        "COLOR_TAB_HOVERED": (0.75, 0.84, 0.98, 1.00),
        "COLOR_TAB_ACTIVE": (0.94, 0.97, 1.00, 1.00),
        "COLOR_NAV_HIGHLIGHT": (0.00, 0.45, 0.95, 0.80),
    }
    for name, rgba in colors.items():
        _set_imgui_color(style, name, rgba)

    io = imgui.get_io()
    fonts = getattr(io, "fonts", None)
    if fonts is None or not hasattr(fonts, "add_font_from_file_ttf"):
        return
    for candidate in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf"),
        Path("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if not candidate.exists():
            continue
        try:
            ranges = fonts.get_glyph_ranges_korean() if hasattr(fonts, "get_glyph_ranges_korean") else None
            font = fonts.add_font_from_file_ttf(str(candidate), 18.0, glyph_ranges=ranges)
            if hasattr(io, "font_default") and font is not None:
                io.font_default = font
            break
        except Exception:
            continue


def _panel_heading(title: str, caption: str = "") -> None:
    if imgui is None:
        return
    imgui.text_colored(title.upper(), 0.05, 0.32, 0.72, 1.0)
    if caption:
        imgui.text_disabled(caption)


def choose_file_with_native_dialog(save: bool, initial: str, extension: str = "") -> Optional[str]:
    """Use a desktop file dialog, falling back through zenity/kdialog/Tk."""
    if shutil.which("zenity"):
        args = ["zenity", "--file-selection"]
        if save:
            args += ["--save", "--confirm-overwrite"]
        result = subprocess.run(args, input=None, capture_output=True, text=True, check=False)
        selected = result.stdout.strip()
        return selected or None
    if shutil.which("kdialog"):
        args = ["kdialog", "--getsavefilename" if save else "--getopenfilename", initial]
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        selected = result.stdout.strip()
        return selected or None
    if tk is not None and filedialog is not None:
        dialog_root = None
        try:
            dialog_root = tk.Tk()
            dialog_root.withdraw()
            dialog_root.attributes("-topmost", True)
            if save:
                selected = filedialog.asksaveasfilename(
                    parent=dialog_root,
                    initialfile=Path(initial).name,
                    initialdir=str(Path(initial).expanduser().parent),
                    defaultextension=extension,
                    filetypes=((f"{extension.lstrip('.').upper()} files", f"*{extension}"), ("All files", "*.*")),
                )
            else:
                selected = filedialog.askopenfilename(
                    parent=dialog_root,
                    initialdir=str(Path(initial).expanduser().parent),
                    filetypes=(("Scenario files", "*.txt"), ("All files", "*.*")),
                )
            return selected.strip() or None
        except Exception:
            return None
        finally:
            if dialog_root is not None:
                try:
                    dialog_root.destroy()
                except Exception:
                    pass
    return None


class MapApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logs: deque[str] = deque(maxlen=600)
        self._log_lock = threading.Lock()
        self.status = "ready"
        self.ports: list[str] = []
        self.port_picker_open_target: Optional[str] = None
        self.motor_port = args.motor_port or self._default_motor_port()
        self.imu_port = args.imu_port or ""
        self.imu_baud = int(args.imu_baud)
        self.motor_position = DEFAULT_POSITION_DEG
        self.motor_velocity = int(args.velocity)
        self.motor_reading = MotorReading()
        self.manual_position = DEFAULT_POSITION_DEG
        self.scenario_path = Path(args.scenario).expanduser()
        self.scenario_name = str(args.scenario_name)
        default_csv = Path(args.csv).expanduser() if args.csv else Path.cwd() / f"map_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
        if default_csv.suffix.lower() == ".xlsx":
            default_csv = default_csv.with_suffix(".csv")
        self.xlsx_path = str(default_csv.with_suffix(".xlsx"))
        self.recorder = CsvRecorder(str(default_csv))
        self.imu = ImuSerialReader(self.log)
        self.motor = DynamixelMotorController(DEFAULT_MOTOR_ID, int(args.baud), self.log)
        self.runner = ScenarioRunner(self.motor, self.log)
        self.last_sample = 0.0
        self.last_motor_error = ""
        self.last_frame_log = 0.0
        self.refresh_ports()

    @staticmethod
    def _default_motor_port() -> str:
        return "COM5" if os.name == "nt" else "/dev/ttyUSB0"

    def log(self, message: str) -> None:
        line = f"[{dt.datetime.now():%H:%M:%S.%f}"[:-3] + f"] {message}"
        with self._log_lock:
            self.logs.append(line)

    def refresh_ports(self) -> None:
        self.ports = []
        if list_ports is not None:
            self.ports = sorted({str(port.device) for port in list_ports.comports()})
        self.log(f"Ports: {', '.join(self.ports) if self.ports else 'none found'}")

    def search_ports(self, target: str) -> None:
        if target not in {"motor", "imu"}:
            raise ValueError(f"unknown port target: {target}")
        self.refresh_ports()
        self.port_picker_open_target = target

    def select_port(self, port_name: str, target: str) -> None:
        if target == "motor":
            self.motor_port = str(port_name)
        elif target == "imu":
            self.imu_port = str(port_name)
        else:
            raise ValueError(f"unknown port target: {target}")
        self.port_picker_open_target = None
        self.log(f"Selected {target} port: {port_name}")

    def sample(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_sample < POLL_PERIOD_SEC:
            return
        self.last_sample = now
        if self.motor.connected:
            self.motor_reading = self.motor.read_state()
            if self.motor_reading.position_deg is not None:
                self.motor_position = self.motor_reading.position_deg
            if self.motor_reading.error and self.motor_reading.error != self.last_motor_error:
                self.last_motor_error = self.motor_reading.error
                self.log(f"Motor: {self.motor_reading.error}")
        imu = self.imu.snapshot()
        row = self._make_row(imu)
        self.recorder.append(row)
        if now - self.last_frame_log >= POLL_PERIOD_SEC:
            self.last_frame_log = now
            self.log(self._format_live_line(imu))

    @staticmethod
    def _csv_value(value: object) -> object:
        return "" if value is None else value

    def _make_row(self, imu: ImuReading) -> dict[str, object]:
        return {
            "timestamp_iso": now_iso(),
            "unix_time": f"{time.time():.6f}",
            "motor_position_deg": self._csv_value(self.motor_reading.position_deg),
            "motor_load": self._csv_value(self.motor_reading.load),
            "imu_roll_deg": self._csv_value(imu.roll if imu.updated_at else None),
            "imu_pitch_deg": self._csv_value(imu.pitch if imu.updated_at else None),
            "imu_yaw_deg": self._csv_value(imu.yaw if imu.updated_at else None),
            "imu_qw": self._csv_value(imu.qw if imu.updated_at else None),
            "imu_qx": self._csv_value(imu.qx if imu.updated_at else None),
            "imu_qy": self._csv_value(imu.qy if imu.updated_at else None),
            "imu_qz": self._csv_value(imu.qz if imu.updated_at else None),
            "imu_magnetometer": self._csv_value(imu.magnetometer),
        }

    def _format_live_line(self, imu: ImuReading) -> str:
        motor = self.motor_reading
        def motor_text(reading: MotorReading) -> str:
            position = "--" if reading.position_deg is None else f"{reading.position_deg:6.2f}"
            load = "--" if reading.load is None else f"{reading.load:6d}"
            return f"pos={position} load={load}"
        def imu_text(reading: ImuReading) -> str:
            if reading.updated_at <= 0.0:
                return "r=-- p=-- y=--"
            return f"r={reading.roll:7.2f} p={reading.pitch:7.2f} y={reading.yaw:7.2f}"
        return f"M(ID {DEFAULT_MOTOR_ID}) {motor_text(motor)} | IMU {imu_text(imu)}"

    # ---------------------------- UI callbacks ---------------------------
    def connect_motor(self) -> None:
        try:
            self.motor.connect(self.motor_port)
            self.status = "motor connected"
            self.sample(force=True)
        except Exception as error:
            self.status = f"motor error: {error}"
            self.log(self.status)

    def disconnect_motor(self) -> None:
        self.runner.stop()
        self.motor.disconnect()
        self.status = "motor disconnected"
        self.log(self.status)

    def connect_imu(self) -> None:
        try:
            self.imu.connect(self.imu_port, self.imu_baud)
            self.status = "IMU connected"
        except Exception as error:
            self.status = f"IMU error: {error}"
            self.log(self.status)

    def disconnect_imu(self) -> None:
        self.imu.disconnect()
        self.status = "IMU disconnected"
        self.log(self.status)

    def send_manual_pose(self) -> None:
        try:
            value = clamp_position(self.manual_position)
            self.motor.set_position(value)
            self.motor_position = value
            self.status = f"manual pose sent: {value:.1f}° (Dynamixel ID {DEFAULT_MOTOR_ID})"
            self.log(self.status)
        except Exception as error:
            self.status = f"manual move error: {error}"
            self.log(self.status)

    def start_scenario(self) -> None:
        try:
            self.runner.start(self.scenario_path, self.scenario_name)
            self.status = f"scenario running: {self.scenario_name}"
        except Exception as error:
            self.status = f"scenario error: {error}"
            self.log(self.status)

    def start_recording(self) -> None:
        self.recorder.start()
        self.status = "recording started"
        self.log(self.status)

    def finish_recording(self) -> None:
        count = self.recorder.finish()
        self.status = f"recording finished: {count} rows"
        self.log(self.status)

    def export_csv(self) -> None:
        try:
            path, count = self.recorder.export()
            self.status = f"CSV exported: {count} rows -> {path}"
            self.log(self.status)
        except Exception as error:
            self.status = f"CSV export error: {error}"
            self.log(self.status)

    def export_xlsx(self) -> None:
        try:
            path, count = self.recorder.export_xlsx(self.xlsx_path)
            self.status = f"XLSX exported: {count} rows -> {path}"
            self.log(self.status)
        except Exception as error:
            self.status = f"XLSX export error: {error}"
            self.log(self.status)

    def browse_csv(self) -> None:
        chosen = choose_file_with_native_dialog(True, self.recorder.path, ".csv")
        if chosen:
            self.recorder.path = chosen
            self.log(f"CSV path: {chosen}")
        else:
            self.log("No desktop file dialog available; edit the CSV path field directly")

    def browse_xlsx(self) -> None:
        chosen = choose_file_with_native_dialog(True, self.xlsx_path, ".xlsx")
        if chosen:
            self.xlsx_path = chosen
            self.log(f"XLSX path: {chosen}")

    def browse_scenario(self) -> None:
        chosen = choose_file_with_native_dialog(False, str(self.scenario_path), ".txt")
        if chosen:
            self.scenario_path = Path(chosen)
            self.log(f"Scenario path: {chosen}")

    def draw(self) -> None:
        if imgui is None:
            return
        self.sample()
        io = imgui.get_io()
        always = getattr(imgui, "ALWAYS", 0)
        imgui.set_next_window_position(0.0, 0.0, always)
        imgui.set_next_window_size(float(io.display_size.x), float(io.display_size.y), always)
        flags = (
            getattr(imgui, "WINDOW_NO_TITLE_BAR", 0)
            | getattr(imgui, "WINDOW_NO_MOVE", 0)
            | getattr(imgui, "WINDOW_NO_RESIZE", 0)
            | getattr(imgui, "WINDOW_NO_COLLAPSE", 0)
        )
        opened = imgui.begin("IMU / MOTOR MAPPER###mapper-root", True, flags=flags)
        if _imgui_open(opened):
            imgui.text_colored("IMU / MOTOR MAPPER", 0.05, 0.32, 0.72, 1.0)
            imgui.same_line()
            imgui.text_disabled("One Dynamixel + one SparkFun BNO080/BNO085")
            imgui.same_line()
            imgui.text_colored(f"STATUS  {self.status}", 0.10, 0.48, 0.22, 1.0)
            imgui.separator()

            available_width = max(1.0, float(imgui.get_content_region_available_width()))
            left_width = min(680.0, max(520.0, available_width * 0.52))
            left_open = imgui.begin_child("mapper-controls", left_width, 0.0, True)
            if _imgui_open(left_open):
                _panel_heading("Control center", "Connections, power, motion and capture")
                imgui.separator()
                self._draw_connections()
                imgui.separator()
                self._draw_manual()
                imgui.separator()
                self._draw_automatic()
                imgui.separator()
                self._draw_recording()
            imgui.end_child()
            imgui.same_line()

            right_open = imgui.begin_child("mapper-telemetry", 0.0, 0.0, True)
            if _imgui_open(right_open):
                self._draw_live_values()
            imgui.end_child()
        imgui.end()

    def _draw_connections(self) -> None:
        def draw_port_selector(label: str, value: str, target: str, suffix: str) -> str:
            changed, value = imgui.input_text(f"{label} port##{suffix}-input", value, 256)
            imgui.same_line()
            if imgui.button(f"Port Search##{suffix}-search", 132.0, 0.0):
                self.search_ports(target)
            if self.port_picker_open_target == target:
                imgui.text_colored(f"Detected ports for {label}:", 0.18, 0.38, 0.68, 1.0)
                if not self.ports:
                    imgui.text_disabled("No serial ports detected")
                for index, port_name in enumerate(self.ports):
                    if imgui.button(f"{port_name}##{suffix}-candidate-{index}", 300.0, 0.0):
                        self.select_port(port_name, target)
                        value = port_name
            return value

        _panel_heading("Dynamixel motor", f"ID {DEFAULT_MOTOR_ID} · one motor")
        self.motor_port = draw_port_selector("Motor", self.motor_port, "motor", "motor-port")
        if imgui.button("Connect motor##connect-motor", 132.0, 0.0):
            self.connect_motor()
        imgui.same_line()
        if imgui.button("Disconnect##disconnect-motor", 132.0, 0.0):
            self.disconnect_motor()
        if imgui.button("Torque ON##torque-on", 132.0, 0.0):
            try:
                self.motor.set_torque(True)
            except Exception as error:
                self.log(f"Torque error: {error}")
        imgui.same_line()
        if imgui.button("Torque OFF##torque-off", 132.0, 0.0):
            try:
                self.motor.set_torque(False)
            except Exception as error:
                self.log(f"Torque error: {error}")

        imgui.separator()
        _panel_heading("Teensy IMU", "one SparkFun BNO080/BNO085")
        self.imu_port = draw_port_selector("IMU", self.imu_port, "imu", "imu-port")
        changed, self.imu_baud = imgui.input_int("IMU baud", self.imu_baud)
        if imgui.button("Connect IMU##connect-imu", 132.0, 0.0):
            self.connect_imu()
        imgui.same_line()
        if imgui.button("Disconnect##disconnect-imu", 132.0, 0.0):
            self.disconnect_imu()
        if imgui.button("IMU ON##imu-on", 132.0, 0.0):
            try:
                self.imu.set_power(True)
            except Exception as error:
                self.log(f"IMU power error: {error}")
        imgui.same_line()
        if imgui.button("IMU OFF##imu-off", 132.0, 0.0):
            try:
                self.imu.set_power(False)
            except Exception as error:
                self.log(f"IMU power error: {error}")
        if imgui.button("Mag ON##mag-on", 132.0, 0.0):
            try:
                self.imu.set_magnetometer(True)
            except Exception as error:
                self.log(f"IMU command error: {error}")
        imgui.same_line()
        if imgui.button("Mag OFF##mag-off", 132.0, 0.0):
            try:
                self.imu.set_magnetometer(False)
            except Exception as error:
                self.log(f"IMU command error: {error}")

    def _draw_manual(self) -> None:
        _panel_heading("Manual movement", f"Dynamixel ID {DEFAULT_MOTOR_ID} target, degrees")
        changed, value = imgui.slider_float(
            f"Motor ID {DEFAULT_MOTOR_ID} position##manual",
            float(self.manual_position), 0.0, 360.0, "%.1f deg"
        )
        if changed:
            self.manual_position = clamp_position(value)
        changed, velocity = imgui.input_int("Profile velocity##manual-velocity", int(self.motor_velocity))
        if changed:
            self.motor_velocity = max(0, min(32767, int(velocity)))
        if imgui.button("Send manual pose##manual-send"):
            try:
                if self.motor.connected:
                    self.motor.set_profile_velocity(self.motor_velocity)
                self.send_manual_pose()
            except Exception as error:
                self.log(f"Manual control error: {error}")
        imgui.same_line()
        if imgui.button("Center##center"):
            self.manual_position = DEFAULT_POSITION_DEG
            self.send_manual_pose()

    def _draw_automatic(self) -> None:
        _panel_heading("Automatic movement", "script.txt scenario DSL")
        changed, raw_path = imgui.input_text("Scenario file##scenario-path", str(self.scenario_path), 512)
        if changed:
            self.scenario_path = Path(raw_path)
        changed, self.scenario_name = imgui.input_text("Scenario name##scenario-name", self.scenario_name, 128)
        if imgui.button("Browse##browse-scenario", 145.0, 0.0):
            self.browse_scenario()
        imgui.same_line()
        if imgui.button("Validate##validate-scenario", 145.0, 0.0):
            try:
                count = len(expand_scenario(self.scenario_path, self.scenario_name))
                self.status = f"scenario valid: {count} commands"
                self.log(self.status)
            except Exception as error:
                self.status = f"scenario invalid: {error}"
                self.log(self.status)
        if imgui.button("Start automatic##start-auto", 145.0, 0.0):
            self.start_scenario()
        imgui.same_line()
        if imgui.button("Stop##stop-auto", 145.0, 0.0):
            self.runner.stop()
            self.status = "automatic movement stopped"
            self.log(self.status)
        imgui.text(f"Runner: {self.runner.status}")

    def _draw_recording(self) -> None:
        _panel_heading("CSV / XLSX capture", "Live rows are kept in memory until export")
        if imgui.button("Start recording##record-start", 145.0, 0.0):
            self.start_recording()
        imgui.same_line()
        if imgui.button("Finish##record-finish", 145.0, 0.0):
            self.finish_recording()
        imgui.text(f"Rows: {len(self.recorder.rows)}  Recording: {'ON' if self.recorder.recording else 'OFF'}")
        imgui.separator()

        changed, path = imgui.input_text("CSV path##csv-path", self.recorder.path, 512)
        if changed:
            self.recorder.path = path
        if imgui.button("Browse CSV##browse-csv", 145.0, 0.0):
            self.browse_csv()
        imgui.same_line()
        if imgui.button("Export CSV##export-csv", 145.0, 0.0):
            self.export_csv()

        changed, path = imgui.input_text("XLSX path##xlsx-path", self.xlsx_path, 512)
        if changed:
            self.xlsx_path = path
        if imgui.button("Browse XLSX##browse-xlsx", 145.0, 0.0):
            self.browse_xlsx()
        imgui.same_line()
        if imgui.button("Export XLSX##export-xlsx", 145.0, 0.0):
            self.export_xlsx()

    def _draw_live_values(self) -> None:
        _panel_heading("Live telemetry", "Latest Dynamixel ID 3 and one IMU reading")
        motor = self.motor_reading
        imu = self.imu.snapshot()
        def motor_text(reading: MotorReading) -> str:
            position = "--" if reading.position_deg is None else f"{reading.position_deg:7.2f}°"
            load = "--" if reading.load is None else f"{reading.load:6d}"
            return f"pos {position}   load {load}"

        def imu_text(reading: ImuReading) -> str:
            if reading.updated_at <= 0.0:
                return "offline"
            magnetometer = "ON" if reading.magnetometer else "OFF"
            return f"R {reading.roll:+7.2f}°   P {reading.pitch:+7.2f}°   Y {reading.yaw:+7.2f}°   MAG {magnetometer}"

        imgui.text_colored(f"MOTOR ID {DEFAULT_MOTOR_ID}", 0.05, 0.32, 0.72, 1.0)
        imgui.same_line()
        imgui.text(motor_text(motor))
        imgui.separator()
        imgui.text_colored("IMU", 0.05, 0.32, 0.72, 1.0)
        imgui.same_line()
        imgui.text(imu_text(imu))
        imgui.separator()
        _panel_heading("Terminal log", "Serial, scenario and recorder events")
        if _imgui_open(imgui.begin_child("live-log", 0.0, 0.0, True)):
            with self._log_lock:
                lines = list(self.logs)
            imgui.text_unformatted("\n".join(lines))
            if getattr(imgui, "get_scroll_y", lambda: 0.0)() >= getattr(imgui, "get_scroll_max_y", lambda: 0.0)() - 5.0:
                imgui.set_scroll_here_y(1.0)
        imgui.end_child()

    def close(self) -> None:
        self.runner.stop()
        self.imu.disconnect()
        self.motor.disconnect()


def run_tk_gui(app: MapApp) -> int:
    """Run the same controls in a native Tk window when GLFW is unavailable."""
    if tk is None or ttk is None:
        print("Tkinter is unavailable; install python3-tk or use a desktop Python", file=sys.stderr)
        app.close()
        return 2
    try:
        root = tk.Tk()
    except Exception as error:
        print(f"Tkinter window could not be created: {error}", file=sys.stderr)
        app.close()
        return 2

    root.title("IMU / MOTOR MAPPER")
    root.geometry("1120x760")
    root.minsize(900, 620)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    motor_port_var = tk.StringVar(value=app.motor_port)
    imu_port_var = tk.StringVar(value=app.imu_port)
    imu_baud_var = tk.StringVar(value=str(app.imu_baud))
    velocity_var = tk.StringVar(value=str(app.motor_velocity))
    position_var = tk.DoubleVar(value=app.manual_position)
    scenario_path_var = tk.StringVar(value=str(app.scenario_path))
    scenario_name_var = tk.StringVar(value=app.scenario_name)
    csv_path_var = tk.StringVar(value=app.recorder.path)
    xlsx_path_var = tk.StringVar(value=app.xlsx_path)
    status_var = tk.StringVar(value=app.status)
    telemetry_var = tk.StringVar(value="waiting for samples")
    runner_var = tk.StringVar(value=app.runner.status)
    rows_var = tk.StringVar(value="Rows: 0  Recording: OFF")
    ports_var = tk.StringVar(value="Detected ports: none")

    def sync_inputs() -> None:
        app.motor_port = motor_port_var.get().strip()
        app.imu_port = imu_port_var.get().strip()
        app.imu_baud = int(imu_baud_var.get())
        app.motor_velocity = max(0, min(32767, int(velocity_var.get())))
        app.manual_position = clamp_position(position_var.get())
        app.scenario_path = Path(scenario_path_var.get()).expanduser()
        app.scenario_name = scenario_name_var.get().strip()
        app.recorder.path = csv_path_var.get().strip()
        app.xlsx_path = xlsx_path_var.get().strip()

    def update_log() -> None:
        with app._log_lock:
            content = "\n".join(app.logs)
        log_widget.configure(state="normal")
        log_widget.delete("1.0", "end")
        log_widget.insert("end", content)
        log_widget.see("end")
        log_widget.configure(state="disabled")

    def update_view() -> None:
        app.sample()
        status_var.set(app.status)
        runner_var.set(f"Runner: {app.runner.status}")
        rows_var.set(
            f"Rows: {len(app.recorder.rows)}  Recording: {'ON' if app.recorder.recording else 'OFF'}"
        )
        motor = app.motor_reading
        position = "--" if motor.position_deg is None else f"{motor.position_deg:.2f}°"
        load = "--" if motor.load is None else str(motor.load)
        imu = app.imu.snapshot()
        if imu.updated_at <= 0.0:
            imu_text = "offline"
        else:
            mag = "ON" if imu.magnetometer else "OFF"
            imu_text = f"R {imu.roll:+.2f}°   P {imu.pitch:+.2f}°   Y {imu.yaw:+.2f}°   MAG {mag}"
        telemetry_var.set(f"Dynamixel ID {DEFAULT_MOTOR_ID}: pos {position}  load {load}\nIMU: {imu_text}")
        ports_var.set(f"Detected ports: {', '.join(app.ports) if app.ports else 'none'}")
        update_log()
        if root.winfo_exists():
            root.after(100, update_view)

    def call(action: Callable[[], None]) -> None:
        try:
            sync_inputs()
            action()
        except Exception as error:
            app.status = f"error: {error}"
            app.log(app.status)
        status_var.set(app.status)

    def search_ports(target: str) -> None:
        try:
            sync_inputs()
            app.search_ports(target)
            render_port_candidates()
        except Exception as error:
            app.status = f"error: {error}"
            app.log(app.status)
            status_var.set(app.status)

    def choose_detected_port(port_name: str, target: str) -> None:
        try:
            sync_inputs()
            app.select_port(port_name, target)
            motor_port_var.set(app.motor_port)
            imu_port_var.set(app.imu_port)
            status_var.set(app.status)
            render_port_candidates()
        except Exception as error:
            app.status = f"error: {error}"
            app.log(app.status)
            status_var.set(app.status)

    def browse_csv() -> None:
        selected = filedialog.asksaveasfilename(
            parent=root, title="Save CSV", initialfile=Path(csv_path_var.get()).name,
            defaultextension=".csv", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if selected:
            csv_path_var.set(selected)
            sync_inputs()

    def browse_xlsx() -> None:
        selected = filedialog.asksaveasfilename(
            parent=root, title="Save XLSX", initialfile=Path(xlsx_path_var.get()).name,
            defaultextension=".xlsx", filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
        if selected:
            xlsx_path_var.set(selected)
            sync_inputs()

    def browse_scenario() -> None:
        selected = filedialog.askopenfilename(
            parent=root, title="Open scenario", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if selected:
            scenario_path_var.set(selected)
            sync_inputs()

    def validate_scenario() -> None:
        count = len(expand_scenario(app.scenario_path, app.scenario_name))
        app.status = f"scenario valid: {count} commands"
        app.log(app.status)

    def send_manual() -> None:
        if app.motor.connected:
            app.motor.set_profile_velocity(app.motor_velocity)
        app.send_manual_pose()

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(0, weight=3, minsize=560)
    main_frame.columnconfigure(1, weight=2, minsize=320)
    main_frame.rowconfigure(1, weight=1)

    title = ttk.Label(main_frame, text="IMU / MOTOR MAPPER", font=("TkDefaultFont", 16, "bold"))
    title.grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(main_frame, textvariable=status_var, foreground="#176b3a").grid(
        row=0, column=1, sticky="e", padx=(12, 0))

    controls = ttk.LabelFrame(main_frame, text="Control center", padding=10)
    controls.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(10, 0))
    telemetry = ttk.LabelFrame(main_frame, text="Live telemetry / log", padding=10)
    telemetry.grid(row=1, column=1, sticky="nsew", pady=(10, 0))
    telemetry.rowconfigure(2, weight=1)
    telemetry.columnconfigure(0, weight=1)

    def field(parent: object, row: int, label: str, variable: object, width: int = 30) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", pady=3)

    def render_port_candidates() -> None:
        for candidate_frame in (motor_port_list_frame, imu_port_list_frame):
            for child in candidate_frame.winfo_children():
                child.destroy()
        target = app.port_picker_open_target
        if target is None:
            return
        candidate_frame = motor_port_list_frame if target == "motor" else imu_port_list_frame
        target_label = "Motor" if target == "motor" else "IMU"
        ttk.Label(
            candidate_frame,
            text=f"Detected ports for {target_label}:",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 3))
        if not app.ports:
            ttk.Label(candidate_frame, text="No serial ports detected").grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(3, 0)
            )
            return
        for index, port_name in enumerate(app.ports, start=1):
            ttk.Button(
                candidate_frame,
                text=port_name,
                command=lambda name=port_name, selected_target=target: choose_detected_port(name, selected_target),
            ).grid(row=index, column=0, columnspan=2, sticky="ew", pady=1)

    controls.columnconfigure(1, weight=1)
    controls.columnconfigure(2, weight=0)
    connections = ttk.LabelFrame(controls, text="Connections", padding=8)
    connections.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    connections.columnconfigure(0, weight=1)
    connections.columnconfigure(1, weight=1)
    connections.columnconfigure(2, weight=0)

    motor_section = ttk.LabelFrame(connections, text=f"Dynamixel motor · ID {DEFAULT_MOTOR_ID}", padding=6)
    motor_section.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
    motor_section.columnconfigure(1, weight=1)
    field(motor_section, 0, "Port", motor_port_var)
    ttk.Button(motor_section, text="Port Search", command=lambda: search_ports("motor")).grid(
        row=0, column=2, sticky="ew", padx=(5, 0), pady=3
    )
    motor_port_list_frame = ttk.Frame(motor_section)
    motor_port_list_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
    motor_port_list_frame.columnconfigure(0, weight=1)
    motor_port_list_frame.columnconfigure(1, weight=1)
    ttk.Button(motor_section, text="Connect", command=lambda: call(app.connect_motor)).grid(
        row=2, column=0, sticky="ew", pady=3
    )
    ttk.Button(motor_section, text="Disconnect", command=lambda: call(app.disconnect_motor)).grid(
        row=2, column=1, sticky="ew", padx=(5, 0), pady=3
    )
    ttk.Button(motor_section, text="Torque ON", command=lambda: call(lambda: app.motor.set_torque(True))).grid(
        row=3, column=0, sticky="ew", pady=3
    )
    ttk.Button(motor_section, text="Torque OFF", command=lambda: call(lambda: app.motor.set_torque(False))).grid(
        row=3, column=1, sticky="ew", padx=(5, 0), pady=3
    )

    imu_section = ttk.LabelFrame(connections, text="Teensy IMU · SparkFun BNO080/BNO085", padding=6)
    imu_section.grid(row=1, column=0, columnspan=3, sticky="ew")
    imu_section.columnconfigure(1, weight=1)
    field(imu_section, 0, "Port", imu_port_var)
    ttk.Button(imu_section, text="Port Search", command=lambda: search_ports("imu")).grid(
        row=0, column=2, sticky="ew", padx=(5, 0), pady=3
    )
    imu_port_list_frame = ttk.Frame(imu_section)
    imu_port_list_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
    imu_port_list_frame.columnconfigure(0, weight=1)
    imu_port_list_frame.columnconfigure(1, weight=1)
    field(imu_section, 2, "Baud", imu_baud_var)
    ttk.Button(imu_section, text="Connect", command=lambda: call(app.connect_imu)).grid(
        row=3, column=0, sticky="ew", pady=3
    )
    ttk.Button(imu_section, text="Disconnect", command=lambda: call(app.disconnect_imu)).grid(
        row=3, column=1, sticky="ew", padx=(5, 0), pady=3
    )
    ttk.Button(imu_section, text="IMU ON", command=lambda: call(lambda: app.imu.set_power(True))).grid(
        row=4, column=0, sticky="ew", pady=3
    )
    ttk.Button(imu_section, text="IMU OFF", command=lambda: call(lambda: app.imu.set_power(False))).grid(
        row=4, column=1, sticky="ew", padx=(5, 0), pady=3
    )
    ttk.Button(imu_section, text="Mag ON", command=lambda: call(lambda: app.imu.set_magnetometer(True))).grid(
        row=5, column=0, sticky="ew", pady=3
    )
    ttk.Button(imu_section, text="Mag OFF", command=lambda: call(lambda: app.imu.set_magnetometer(False))).grid(
        row=5, column=1, sticky="ew", padx=(5, 0), pady=3
    )
    ttk.Label(connections, textvariable=ports_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))
    render_port_candidates()
    ttk.Label(controls, text=f"Position (ID {DEFAULT_MOTOR_ID})").grid(row=13, column=0, columnspan=2, sticky="w")
    tk.Scale(controls, from_=0, to=360, resolution=0.1, orient="horizontal", variable=position_var, showvalue=True).grid(
        row=14, column=0, columnspan=2, sticky="ew")
    field(controls, 15, "Profile velocity", velocity_var)
    ttk.Button(controls, text="Send position", command=lambda: call(send_manual)).grid(row=16, column=0, sticky="ew", pady=3)
    ttk.Button(controls, text="Center", command=lambda: call(lambda: (position_var.set(DEFAULT_POSITION_DEG), send_manual()))).grid(row=16, column=1, sticky="ew", pady=3)
    ttk.Separator(controls).grid(row=17, column=0, columnspan=2, sticky="ew", pady=8)
    field(controls, 18, "Scenario file", scenario_path_var)
    ttk.Button(controls, text="Browse scenario", command=browse_scenario).grid(row=19, column=0, sticky="ew", pady=3)
    field(controls, 20, "Scenario name", scenario_name_var)
    ttk.Button(controls, text="Validate", command=lambda: call(validate_scenario)).grid(row=21, column=0, sticky="ew", pady=3)
    ttk.Button(controls, text="Start automatic", command=lambda: call(app.start_scenario)).grid(row=21, column=1, sticky="ew", pady=3)
    ttk.Button(controls, text="Stop", command=lambda: call(app.runner.stop)).grid(row=22, column=0, sticky="ew", pady=3)
    ttk.Label(controls, textvariable=runner_var).grid(row=22, column=1, sticky="w")
    capture = ttk.LabelFrame(controls, text="CSV / XLSX capture", padding=6)
    capture.grid(row=23, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    capture.columnconfigure(1, weight=1)
    ttk.Label(capture, text="Recording controls").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 3)
    )
    ttk.Button(capture, text="Start recording", command=lambda: call(app.start_recording)).grid(
        row=1, column=0, sticky="ew", pady=3
    )
    ttk.Button(capture, text="Finish recording", command=lambda: call(app.finish_recording)).grid(
        row=1, column=1, sticky="ew", padx=(5, 0), pady=3
    )
    ttk.Label(capture, textvariable=rows_var).grid(
        row=2, column=0, columnspan=2, sticky="w", pady=(2, 4)
    )
    ttk.Separator(capture).grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
    field(capture, 4, "CSV path", csv_path_var)
    ttk.Button(capture, text="Browse CSV", command=browse_csv).grid(
        row=5, column=0, sticky="ew", pady=3
    )
    ttk.Button(capture, text="Export CSV", command=lambda: call(app.export_csv)).grid(
        row=5, column=1, sticky="ew", padx=(5, 0), pady=3
    )
    field(capture, 6, "XLSX path", xlsx_path_var)
    ttk.Button(capture, text="Browse XLSX", command=browse_xlsx).grid(
        row=7, column=0, sticky="ew", pady=3
    )
    ttk.Button(capture, text="Export XLSX", command=lambda: call(app.export_xlsx)).grid(
        row=7, column=1, sticky="ew", padx=(5, 0), pady=3
    )

    ttk.Label(telemetry, textvariable=telemetry_var, justify="left", font=("TkFixedFont", 11)).grid(
        row=0, column=0, sticky="nw")
    ttk.Label(telemetry, text="Serial, scenario and recorder events").grid(row=1, column=0, sticky="w", pady=(10, 4))
    log_widget = tk.Text(telemetry, height=20, wrap="none", state="disabled", font=("TkFixedFont", 9))
    log_widget.grid(row=2, column=0, sticky="nsew")
    log_scroll = ttk.Scrollbar(telemetry, orient="vertical", command=log_widget.yview)
    log_scroll.grid(row=2, column=1, sticky="ns")
    log_widget.configure(yscrollcommand=log_scroll.set)

    closed = False

    def close_window() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(0, update_view)
    try:
        root.mainloop()
    finally:
        if not closed:
            close_window()
    return 0


def _run_glfw_gui(app: MapApp) -> int:
    if glfw is None or imgui is None or GlfwRenderer is None:
        raise RuntimeError(f"GLFW/ImGui unavailable: {GUI_IMPORT_ERROR or 'dependency not installed'}")
    # WSLg commonly exports both WAYLAND_DISPLAY and DISPLAY.  Prefer X11
    # when DISPLAY is available because it gives PyOpenGL the GLX context
    # identity expected by pyimgui's renderer.
    if (
        os.environ.get("DISPLAY")
        and hasattr(glfw, "init_hint")
        and hasattr(glfw, "PLATFORM")
        and hasattr(glfw, "PLATFORM_X11")
    ):
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    if hasattr(glfw, "set_error_callback"):
        glfw.set_error_callback(
            lambda code, description: print(
                f"GLFW error {code}: {description}", file=sys.stderr
            )
        )
    initialized = False
    window = None
    renderer = None
    try:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        initialized = True
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        if hasattr(glfw, "CONTEXT_CREATION_API") and hasattr(glfw, "NATIVE_CONTEXT_API"):
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        if hasattr(glfw, "OPENGL_PROFILE") and hasattr(glfw, "OPENGL_CORE_PROFILE"):
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(1280, 900, "IMU / MOTOR MAPPER", None, None)
        if not window:
            raise RuntimeError("cannot create GLFW window")
        glfw.make_context_current(window)
        if hasattr(glfw, "get_current_context") and glfw.get_current_context() is None:
            raise RuntimeError("GLFW created a window but no OpenGL context is current")
        glfw.swap_interval(1)
        imgui.create_context()
        _install_imgui_theme()
        renderer = GlfwRenderer(window)
        while not glfw.window_should_close(window):
            glfw.poll_events()
            renderer.process_inputs()
            imgui.new_frame()
            app.draw()
            imgui.render()
            renderer.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
        return 0
    finally:
        if renderer is not None:
            renderer.shutdown()
        if window is not None:
            glfw.destroy_window(window)
        if initialized:
            glfw.terminate()


def run_gui(app: MapApp) -> int:
    try:
        result = _run_glfw_gui(app)
        app.close()
        return result
    except Exception as error:
        print(f"GLFW/ImGui unavailable ({error}); switching to native Tkinter UI", file=sys.stderr)
        return run_tk_gui(app)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GLFW/Tkinter single-Dynamixel and single-SparkFun-IMU mapper")
    parser.add_argument("--motor-port", default="", help="Dynamixel serial port")
    parser.add_argument("--imu-port", default="", help="Teensy IMU serial port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--imu-baud", type=int, default=DEFAULT_IMU_BAUDRATE)
    parser.add_argument("--velocity", type=int, default=50)
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO_FILE))
    parser.add_argument("--scenario-name", default="cycle_upward")
    parser.add_argument("--csv", default="", help="initial CSV export path")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = MapApp(args)
    try:
        return run_gui(app)
    except Exception as error:
        app.close()
        print(f"map.py GUI startup failed: {error}", file=sys.stderr)
        print(
            "Linux/WSLg hint: ensure DISPLAY is valid; "
            "the program selects X11 + GLX automatically when DISPLAY exists.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
