# New IMU / motor / CV tools

## Install

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.\\.venv\\Scripts\\Activate.ps1` and run `python map.py` or `python cv.py`.
Python 3.12+ installs Dear ImGui Bundle; Python 3.14 is supported through the
conditional dependency in `requirements.txt`.

## Mapper

```bash
python map.py --motor-port COM5 --imu-port COM6
```

`map.py` uses Dynamixel Protocol 2.0 with one motor (ID 3), and reads the
single SparkFun BNO080/BNO085 CSV protocol emitted by `imu.ino`.  The GUI first
tries GLFW/ImGui and falls back to a native Tkinter window.  `MAG ON` and
`MAG OFF` switch the magnetometer-backed rotation vector on the Teensy;
`IMU ON`/`IMU OFF` control the BNO sleep state.  The automatic runner accepts
the existing `script.txt` sections (`move`, `pause`, `velocity`, `run`, `mark`,
`cycle`, and `seqmove`).  Captures can be exported as both CSV and XLSX.

## CV tracker

```bash
python cv.py
```

No arguments opens a GLFW/ImGui input wizard (Tkinter fallback) for the video file, CSV output,
slave count, and Auto/Manual session mode.  The first frame then asks for a
base click followed by ordered slave clicks.  The tracking session starts
paused at frame 1.  Auto detects while playing; Manual lets the user pause and
press Detect for individual frames.  Later frames are accepted only when
colour, one-to-one assignment, motion range, chain geometry, and segment
length are all plausible.  CLI use is also supported with
`input.mp4 --mode manual --csv tracked.csv`.
