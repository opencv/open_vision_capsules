import warnings
from typing import Optional
from pkg_resources import parse_version

import functools

from . import __version__


def deprecated(message: str = "",
               remove_in: Optional[str] = None,
               current_version=__version__):
    """Mark a function as deprecated
    :param message: Extra details to be added to the warning. For example,
    the details may point users to a replacement method.
    :param remove_in: The version when the decorated method will be removed.
    The default is None, specifying that the function is not currently planned
    to be removed.
    :param current_version: Defaults to the vcaps current version.
    """

    def wrapper(function):
        warning_msg = f"'{function.__qualname__}' is deprecated "
        warning_msg += (f"and is scheduled to be removed in {remove_in}. "
                        if remove_in is not None else ". ")
        warning_msg += str(message)

        # Only show the DeprecationWarning on the first call
        warnings.simplefilter("once", DeprecationWarning, append=True)

        @functools.wraps(function)
        def inner(*args, **kwargs):
            if remove_in is not None and \
                    parse_version(current_version) >= parse_version(remove_in):
                raise DeprecationWarning(warning_msg)
            else:
                warnings.warn(warning_msg, category=DeprecationWarning)
            return function(*args, **kwargs)

        return inner

    return wrapper
