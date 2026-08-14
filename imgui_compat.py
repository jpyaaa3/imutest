"""Small compatibility layer for pyimgui and Dear ImGui Bundle.

The original project used pyimgui's legacy GLFW integration.  pyimgui 2.0
does not publish a CPython 3.14 wheel, while Dear ImGui Bundle does.  This
module keeps the handful of old call signatures used by map.py and cv.py
working with either backend.
"""

from __future__ import annotations

from typing import Any, Optional


try:
    from imgui_bundle import imgui as _backend
    from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

    USING_IMGUI_BUNDLE = True
except Exception:
    import imgui as _backend  # type: ignore[no-redef]
    from imgui.integrations.glfw import GlfwRenderer  # type: ignore[no-redef]

    USING_IMGUI_BUNDLE = False


def _enum_value(enum_name: str, value: Any) -> Any:
    """Convert an integer flag/condition to Dear ImGui Bundle's enum type."""
    if not USING_IMGUI_BUNDLE:
        return value
    enum_type = getattr(_backend, enum_name, None)
    if enum_type is None or not isinstance(value, int):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return value


class _ColorProxy:
    def __init__(self, style: Any) -> None:
        self._style = style

    def __setitem__(self, index: int, color: Any) -> None:
        if hasattr(self._style, "set_color_"):
            self._style.set_color_(int(index), color)
            return
        self._style.colors[index] = color

    def __getitem__(self, index: int) -> Any:
        if hasattr(self._style, "color_"):
            return self._style.color_(int(index))
        return self._style.colors[index]


class _StyleProxy:
    """Make Bundle's ImVec2 style fields accept the old tuple assignments."""

    def __init__(self, style: Any) -> None:
        object.__setattr__(self, "_style", style)
        object.__setattr__(self, "colors", _ColorProxy(style))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._style, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_style", "colors"}:
            object.__setattr__(self, name, value)
            return
        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                current = getattr(self._style, name)
                if hasattr(current, "x") and hasattr(current, "y"):
                    current.x = float(value[0])
                    current.y = float(value[1])
                    return
            except Exception:
                pass
        setattr(self._style, name, value)


class _DrawListProxy:
    def __init__(self, draw_list: Any) -> None:
        self._draw_list = draw_list

    def add_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: int,
        thickness: float = 1.0,
    ) -> None:
        if USING_IMGUI_BUNDLE:
            self._draw_list.add_line((x1, y1), (x2, y2), color, thickness)
        else:
            self._draw_list.add_line(x1, y1, x2, y2, color, thickness)

    def add_text(self, x: float, y: float, color: int, text: str) -> None:
        if USING_IMGUI_BUNDLE:
            self._draw_list.add_text((x, y), color, text)
        else:
            self._draw_list.add_text(x, y, color, text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._draw_list, name)


_FLAG_ALIASES = {
    "WINDOW_NO_TITLE_BAR": ("WindowFlags_", "no_title_bar"),
    "WINDOW_NO_MOVE": ("WindowFlags_", "no_move"),
    "WINDOW_NO_RESIZE": ("WindowFlags_", "no_resize"),
    "WINDOW_NO_COLLAPSE": ("WindowFlags_", "no_collapse"),
}

_COLOR_ALIASES = {
    "COLOR_TEXT": "text",
    "COLOR_TEXT_DISABLED": "text_disabled",
    "COLOR_WINDOW_BACKGROUND": "window_bg",
    "COLOR_CHILD_BACKGROUND": "child_bg",
    "COLOR_POPUP_BACKGROUND": "popup_bg",
    "COLOR_BORDER": "border",
    "COLOR_BORDER_SHADOW": "border_shadow",
    "COLOR_FRAME_BACKGROUND": "frame_bg",
    "COLOR_FRAME_BACKGROUND_HOVERED": "frame_bg_hovered",
    "COLOR_FRAME_BACKGROUND_ACTIVE": "frame_bg_active",
    "COLOR_TITLE_BACKGROUND": "title_bg",
    "COLOR_TITLE_BACKGROUND_ACTIVE": "title_bg_active",
    "COLOR_TITLE_BACKGROUND_COLLAPSED": "title_bg_collapsed",
    "COLOR_MENU_BAR_BACKGROUND": "menu_bar_bg",
    "COLOR_SCROLLBAR_BACKGROUND": "scrollbar_bg",
    "COLOR_SCROLLBAR_GRAB": "scrollbar_grab",
    "COLOR_SCROLLBAR_GRAB_HOVERED": "scrollbar_grab_hovered",
    "COLOR_SCROLLBAR_GRAB_ACTIVE": "scrollbar_grab_active",
    "COLOR_CHECK_MARK": "check_mark",
    "COLOR_SLIDER_GRAB": "slider_grab",
    "COLOR_SLIDER_GRAB_ACTIVE": "slider_grab_active",
    "COLOR_BUTTON": "button",
    "COLOR_BUTTON_HOVERED": "button_hovered",
    "COLOR_BUTTON_ACTIVE": "button_active",
    "COLOR_HEADER": "header",
    "COLOR_HEADER_HOVERED": "header_hovered",
    "COLOR_HEADER_ACTIVE": "header_active",
    "COLOR_SEPARATOR": "separator",
    "COLOR_SEPARATOR_HOVERED": "separator_hovered",
    "COLOR_SEPARATOR_ACTIVE": "separator_active",
    "COLOR_TAB": "tab",
    "COLOR_TAB_HOVERED": "tab_hovered",
    "COLOR_TAB_ACTIVE": "tab_active",
    "COLOR_NAV_HIGHLIGHT": "nav_highlight",
}

_KEY_ALIASES = {
    "KEY_ESCAPE": "escape",
    "KEY_BACKSPACE": "backspace",
    "KEY_ENTER": "enter",
}


class _ImGuiCompat:
    def __getattr__(self, name: str) -> Any:
        if USING_IMGUI_BUNDLE:
            if name in _FLAG_ALIASES:
                enum_name, member_name = _FLAG_ALIASES[name]
                return getattr(getattr(_backend, enum_name), member_name)
            if name == "ALWAYS":
                return getattr(_backend.Cond_, "always")
            if name == "ONCE":
                return getattr(_backend.Cond_, "once")
            if name in _KEY_ALIASES:
                return getattr(_backend.Key, _KEY_ALIASES[name])
            if name in _COLOR_ALIASES:
                return getattr(_backend.Col_, _COLOR_ALIASES[name]).value
        return getattr(_backend, name)

    def get_style(self) -> _StyleProxy:
        return _StyleProxy(_backend.get_style())

    def begin(self, name: str, opened: Any = True, flags: Any = 0) -> Any:
        if USING_IMGUI_BUNDLE:
            return _backend.begin(name, opened, _enum_value("WindowFlags_", flags))
        return _backend.begin(name, opened, flags=flags)

    def begin_child(
        self,
        identifier: str,
        width: float = 0.0,
        height: float = 0.0,
        border: bool = False,
        flags: Any = 0,
    ) -> Any:
        if USING_IMGUI_BUNDLE:
            child_flags = getattr(_backend.ChildFlags_, "borders") if border else 0
            return _backend.begin_child(
                identifier,
                (float(width), float(height)),
                child_flags,
                _enum_value("WindowFlags_", flags),
            )
        return _backend.begin_child(identifier, width, height, border, flags)

    def button(self, label: str, width: float = 0.0, height: float = 0.0) -> bool:
        if USING_IMGUI_BUNDLE:
            return _backend.button(label, (float(width), float(height)))
        return _backend.button(label, width, height)

    def image(self, texture: Any, width: float, height: float) -> None:
        if USING_IMGUI_BUNDLE:
            texture_ref = texture
            if isinstance(texture, int) and hasattr(_backend, "ImTextureRef"):
                texture_ref = _backend.ImTextureRef(texture)
            _backend.image(texture_ref, (float(width), float(height)))
        else:
            _backend.image(texture, width, height)

    def input_text(self, label: str, value: str, buffer_size: int = 0, *args: Any, **kwargs: Any) -> Any:
        if USING_IMGUI_BUNDLE:
            return _backend.input_text(label, value)
        return _backend.input_text(label, value, buffer_size, *args, **kwargs)

    def text_colored(self, text: str, red: float, green: float, blue: float, alpha: float) -> None:
        if USING_IMGUI_BUNDLE:
            _backend.text_colored((red, green, blue, alpha), text)
        else:
            _backend.text_colored(text, red, green, blue, alpha)

    def set_next_window_position(self, x: float, y: float, condition: Any = 0) -> None:
        if USING_IMGUI_BUNDLE:
            _backend.set_next_window_pos((float(x), float(y)), _enum_value("Cond_", condition))
        else:
            _backend.set_next_window_position(x, y, condition)

    def set_next_window_size(self, width: float, height: float, condition: Any = 0) -> None:
        if USING_IMGUI_BUNDLE:
            _backend.set_next_window_size((float(width), float(height)), _enum_value("Cond_", condition))
        else:
            _backend.set_next_window_size(width, height, condition)

    def get_content_region_available_width(self) -> float:
        if USING_IMGUI_BUNDLE:
            return float(_backend.get_content_region_avail().x)
        return float(_backend.get_content_region_available_width())

    def get_color_u32_rgba(self, red: float, green: float, blue: float, alpha: float) -> int:
        if USING_IMGUI_BUNDLE:
            return int(_backend.get_color_u32((red, green, blue, alpha)))
        return int(_backend.get_color_u32_rgba(red, green, blue, alpha))

    def get_window_draw_list(self) -> _DrawListProxy:
        return _DrawListProxy(_backend.get_window_draw_list())


imgui = _ImGuiCompat()


__all__ = ["GlfwRenderer", "USING_IMGUI_BUNDLE", "imgui"]
