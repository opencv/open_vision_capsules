import os
import readline
import sys
from enum import Enum
from pathlib import Path

import i18n


class Color(Enum):
    """ANSI escape codes representing colors in the terminal theme."""

    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_color(message, color: Color, **kwargs) -> None:
    color = _check_no_color(color)
    print(f"{color.value}{message}{Color.END.value}", **kwargs)


def _check_no_color(color: Color) -> Color:
    """Turns off color if the user wants.
    See: https://no-color.org/
    """
    if "NO_COLOR" in os.environ:
        return Color.END
    return color

