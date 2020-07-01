from abc import ABC, abstractmethod
from typing import List, Optional, Union


OPTION_TYPE = Union[int, float, bool, str]


def _check_number_range(value, min_val, max_val):
    error = ValueError(f"Value {value} is not in "
                       f"{min_val} <= x <= {max_val}")

    if min_val is not None and value < min_val:
        raise error
    if max_val is not None and value > max_val:
        raise error


class Option(ABC):
    """A class that all options subclass. Added mostly for documentation an
    type hinting purposes.
    """

    def __init__(self, default, description):
        self.default = default
        self.description: Optional[str] = description
        self.check(default)

    @abstractmethod
    def check(self, value):
        """Checks the given value to see if it fits this option.

        :param value: The value to check
        :raises ValueError: If the value does not fit the option
        """
        pass


class FloatOption(Option):
    """A capsule option that holds a floating point value with defined
    boundaries.
    """

    def __init__(self, *, default: float,
                 min_val: Optional[float],
                 max_val: Optional[float],
                 description: Optional[str] = None):
        """
        :param default: The default value of this option
        :param min_val: The minimum allowed value for this option, inclusive,
            or None for no lower limit
        :param max_val: The maximum allowed value for this option, inclusive,
            or None for no upper limit
        :param description: The description for this option
        """
        self.min_val = min_val
        self.max_val = max_val

        super().__init__(default=default, description=description)

    def check(self, value):
        if not isinstance(value, float):
            raise ValueError(f"Expecting type float, got {type(value)}")
        _check_number_range(value, self.min_val, self.max_val)


class IntOption(Option):
    """A capsule option that holds an integer value.
    """

    def __init__(self, *, default: int,
                 min_val: Optional[int],
                 max_val: Optional[int],
                 description: Optional[str] = None):
        """
        :param default: The default value of this option
        :param min_val: The minimum allowed value for this option, inclusive,
            or None for no lower limit
        :param max_val: The maximum allowed value for this option, inclusive,
            or None for no upper limit
        :param description: The description for this option
        """
        self.min_val = min_val
        self.max_val = max_val

        super().__init__(default=default, description=description)

    def check(self, value):
        if not isinstance(value, int):
            raise ValueError(f"Expecting type int, got {type(value)}")
        _check_number_range(value, self.min_val, self.max_val)


class EnumOption(Option):
    """A capsule option that holds a choice from a discrete set of string
    values.
    """

    def __init__(self, *, default: str, choices: List[str],
                 description: Optional[str] = None):
        """
        :param default: The default value of this option
        :param choices: A list of all valid values for this option
        :param description: The description for this option
        """
        assert len(choices) > 0

        self.choices = choices

        super().__init__(default=default, description=description)

    def check(self, value):
        if not isinstance(value, str):
            raise ValueError(f"Expecting type float, got {type(value)}")

        if value not in self.choices:
            raise ValueError(f"Value {value} is not one of {self.choices}")


class BoolOption(Option):
    """A capsule option that holds an boolean value."""

    def __init__(self, *, default: bool,
                 description: Optional[str] = None):
        """
        :param default: The default value of this option
        :param description: The description for this option
        """
        super().__init__(default=default, description=description)

    def check(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Expecting type bool, got {type(value)}")


def check_option_values(capsule, option_vals):
    """Checks the given option values against the capsule's available options to
    make sure they are of the right name, type, and fit within the constraints.
    """
    for name, val in option_vals.items():
        if name not in capsule.options:
            raise ValueError(f"'{name}' is not a valid option for this capsule")
        option = capsule.options[name]
        try:
            option.check(val)
        except ValueError as e:
            raise ValueError(f"For option {name}: {e}")


"""Options that detector capsules tend to have
to avoid breaking API too often, make these options be short, with minimal 
keys. Try to add a name that is descriptive to the use case of the capsule.

If an existing one is changed, it is the responsibility of the person
changing to verify that all capsules have been updated. This is considered
a breaking capsule API change if the key is changed.
"""

common_detector_options = {
    "threshold": FloatOption(
        default=0.5,
        min_val=0.0,
        max_val=1.0,
        description="The confidence threshold for the detector to return a "
                    "detection for an object. Lower => more detections, "
                    "higher means fewer but more accurate detections."),
}
