import configparser
import hashlib
import re
import sys
from importlib.machinery import ModuleSpec
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, List, Optional, Union
from zipfile import ZipFile

from vcap import BaseCapsule, BaseBackend, BaseStreamState, NodeDescription
from vcap.loading.errors import IncompatibleCapsuleError, InvalidCapsuleError

from vcap.loading.crypto_utils import decrypt
from vcap.loading.import_hacks import ZipFinder
from vcap.loading.vcap_packaging import CAPSULE_FILE_NAME, META_FILE_NAME

MAJOR_COMPATIBLE_VERSION = 0
MINOR_COMPATIBLE_VERSION = 3
"""The capsule version that this version of the library supports."""

MAJOR_MINOR_SEMVER_PATTERN = re.compile(r"([0-9]+)\.([0-9]+)")


def load_capsule_from_bytes(data: bytes,
                            source_path: Optional[Path] = None,
                            key: Optional[str] = None,
                            inference_mode: bool = True) -> BaseCapsule:
    """Loads a capsule from the given bytes.

    :param data: The data of the capsule
    :param source_path: The path to the capsule's source code, if it's
        available at runtime
    :param key: The AES key to decrypt the capsule with, or None if the capsule
        is not encrypted
    :param inference_mode: If True, the backends for this capsule will be
        started. If False, the capsule will never be able to run inference, but
        it will still have it's various readable attributes.
    :return: The loaded capsule object
    """
    module_name = capsule_module_name(data)
    if source_path is None:
        # The source is unavailable. Use a dummy path
        source_path = Path("/", module_name)

    if key is not None:
        # Decrypt the capsule into its original form, a zip file
        data = decrypt(key, data)

    file_like = BytesIO(data)
    loaded_files = {}

    code = None
    with ZipFile(file_like, "r") as capsule_file:
        if CAPSULE_FILE_NAME not in capsule_file.namelist():
            raise InvalidCapsuleError(f"Missing file {CAPSULE_FILE_NAME}")

        if META_FILE_NAME not in capsule_file.namelist():
            raise InvalidCapsuleError(f"Missing file {META_FILE_NAME}")

        for name in capsule_file.namelist():
            if name == CAPSULE_FILE_NAME:
                # Every capsule has a capsule.py file defining the capsule's
                # behavior
                code = capsule_file.read(CAPSULE_FILE_NAME)
            else:
                # Load all other files as well
                loaded_files[name] = capsule_file.read(name)

        # Read the meta.conf and get the OpenVisionCapsules API compatibility
        # version
        meta_conf = configparser.ConfigParser()
        meta_conf.read_string(loaded_files[META_FILE_NAME].decode("utf-8"))
        compatibility_version = meta_conf["about"]["api_compatibility_version"]

        match = MAJOR_MINOR_SEMVER_PATTERN.fullmatch(compatibility_version)
        if match is None:
            raise InvalidCapsuleError(
                f"Invalid API compatibility version format "
                f"'{compatibility_version}'. Version must be in the format "
                f"'[major].[minor]'.")
        try:
            major, minor = map(int, (match[1], match[2]))
        except ValueError:
            raise InvalidCapsuleError(
                f"Compatibility versions must be numbers, got "
                f"{major}.{minor}.")
        if major != MAJOR_COMPATIBLE_VERSION:
            raise IncompatibleCapsuleError(
                f"The capsule is not compatible with this software. The "
                f"capsule's OpenVisionCapsules required major version is "
                f"{major} but this software uses OpenVisionCapsules "
                f"{MAJOR_COMPATIBLE_VERSION}.{MINOR_COMPATIBLE_VERSION}.")
        elif (MAJOR_COMPATIBLE_VERSION == 0
              and minor != MINOR_COMPATIBLE_VERSION):
            # Because vcap has not yet reached a 1.0 API, while the major
            # version is 0 then minor version will be required to match.
            raise IncompatibleCapsuleError(
                "The capsule is not compatible with this software. The "
                "capsule's OpenVisionCapsules required version is "
                f"{major}.{minor}, but this software uses OpenVisionCapsules "
                f"{MAJOR_COMPATIBLE_VERSION}.{MINOR_COMPATIBLE_VERSION}"
            )

        if minor > MINOR_COMPATIBLE_VERSION:
            raise IncompatibleCapsuleError(
                f"The capsule requires a version of OpenVisionCapsules "
                f"that is too new for this software. The capsule requires at "
                f"least version {major}.{minor} but this software uses "
                f"OpenVisionCapsules "
                f"{MAJOR_COMPATIBLE_VERSION}.{MINOR_COMPATIBLE_VERSION}.")

        # With the capsule's code loaded, initialize the object
        capsule_module = ModuleType(module_name)
        # Put the module in a package so that it can do relative imports
        capsule_module.__package__ = module_name

        # Allow the capsule.py to import other files in the capsule
        zip_finder = ZipFinder(capsule_file, source_path, module_name)
        sys.meta_path.insert(1, zip_finder)

        try:
            # Run the capsule
            compiled = compile(code, source_path / CAPSULE_FILE_NAME, "exec")
            exec(compiled, capsule_module.__dict__)
        except Exception as e:
            raise InvalidCapsuleError(
                f"Could not execute the code in the capsule!\n"
                f"Error: {e}")
        finally:
            # Remove custom import code
            sys.meta_path.remove(zip_finder)

    # noinspection PyUnresolvedReferences
    new_capsule: BaseCapsule = capsule_module.Capsule(
        capsule_files=loaded_files,
        inference_mode=inference_mode)

    try:
        _validate_capsule(new_capsule)
    except InvalidCapsuleError as e:
        new_capsule.close()
        raise e

    return new_capsule


def load_capsule(path: Union[str, Path],
                 source_path: Optional[Path] = None,
                 key: Optional[str] = None,
                 inference_mode: bool = True) -> BaseCapsule:
    """Load a capsule from the filesystem.

    :param path: The path to the capsule file
    :param source_path: The path to the capsule's source code, if it's
        available at runtime
    :param key: The AES key to decrypt the capsule with, or None if the capsule
        is not encrypted
    :param inference_mode: If True, the backends for this capsule will be
        started. If False, the capsule will never be able to run inference, but
        it will still have it's various readable attributes.
    """
    path = Path(path)

    if source_path is None:
        # Set the default source path to a directory alongside the capsule file
        source_path = path.absolute().with_suffix("")

    return load_capsule_from_bytes(
        data=path.read_bytes(),
        source_path=source_path,
        key=key,
        inference_mode=inference_mode,
    )


def _validate_capsule(capsule: BaseCapsule):
    """This will try calling different attributes on a capsule to make sure
    they are there. This is based on the API compatibility version.

    :raises: InvalidCapsuleError
    """

    def check_arg_names(func: Callable,
                        correct: List[str],
                        ignore: Optional[List[str]] = None) \
            -> bool:
        """Return False if a function has the wrong argument names. Return
        true if they are correct.
        Usage:
        >>> def my_func(self, frame, detection_node):
        ...     pass
        >>> check_arg_names(my_func, ['self'], ['frame', 'detection_node'])
        True
        """
        # noinspection PyUnresolvedReferences
        code = func.__code__

        ignore = [] if ignore is None else ignore
        all_var_names = code.co_varnames
        arg_names = all_var_names[:code.co_argcount]
        filtered = [n for n in arg_names if n not in ignore]
        return filtered == correct

    try:
        # If the ASCII flag is used, only [a-zA-Z0-9_] is matched for \w.
        # Otherwise, some other Unicode characters can be matched, depending on
        # the locale. We don't want that.
        if re.fullmatch(r"\w+", capsule.name, flags=re.ASCII) is None:
            raise InvalidCapsuleError(
                "Capsule names must only contain alphanumeric characters and "
                "underscores",
                capsule.name)

        # Validate the capsule class attributes
        _validate_capsule_field(capsule, "name", capsule.name, str)
        _validate_capsule_field(capsule,
                                "backend_loader",
                                capsule.backend_loader,
                                callable)
        _validate_capsule_field(capsule, "version", capsule.version, int)
        _validate_capsule_field(capsule,
                                "input_type",
                                capsule.input_type,
                                NodeDescription)
        _validate_capsule_field(capsule,
                                "output_type",
                                capsule.input_type,
                                NodeDescription)
        _validate_capsule_field(capsule, "options", capsule.options, dict)

        # Make sure that certain things are NOT attributes (we don't want
        # accidental leftover code from previous capsule versions)
        unwanted_attributes = ["backend_config"]
        for unwanted_attr in unwanted_attributes:
            try:
                # This should throw an attribute error
                capsule.__getattribute__(unwanted_attr)
                raise InvalidCapsuleError(
                    f"The capsule has leftover attributes from a previous "
                    f"OpenVisionCapsules API version. Attribute name: "
                    f"{unwanted_attr}",
                    capsule.name)
            except AttributeError:
                pass

        # Validate the capsule's backend_loader takes the right args
        correct_args = ["capsule_files", "device"]
        is_correct = check_arg_names(func=capsule.backend_loader,
                                     correct=correct_args)
        if not is_correct:
            raise InvalidCapsuleError(
                f"The capsule's backend_loader has invalid arguments!\n"
                f"The arguments must be {correct_args}",
                capsule.name)

        # Validate the backend class attributes
        if capsule.backends is not None:
            backend = capsule.backends[0]
            _validate_backend_field(capsule,
                                    "batch_predict",
                                    backend.batch_predict,
                                    callable)
            _validate_backend_field(capsule,
                                    "process_frame",
                                    backend.process_frame,
                                    callable)
            _validate_backend_field(capsule,
                                    "close",
                                    backend.close,
                                    callable)
            if not isinstance(capsule.backends[0], BaseBackend):
                raise InvalidCapsuleError(
                    f"The capsule's backend field must be a class that "
                    f"subclasses {BaseBackend.__name__}",
                    capsule.name)

        # Validate the stream state
        stream_state = capsule.stream_state
        if not issubclass(stream_state, BaseStreamState):
            raise InvalidCapsuleError(
                f"The capsule's stream_state field must be a subclass of "
                f"{BaseStreamState.__name__}, got {stream_state}")

        # Validate that if the capsule is an encoder, it has a threshold option
        if capsule.capability.encoded:
            if "recognition_threshold" not in capsule.options.keys():
                raise InvalidCapsuleError(
                    "This capsule can encode, but doesn't have a option "
                    "called \"recognition_threshold\"!",
                    capsule.name)
    except InvalidCapsuleError:
        # Don't catch this exception type in the "except Exception" below
        raise
    except AttributeError as e:
        message = f"The capsule did not describe the necessary attributes. " \
                  f"Error: {e}"
        raise InvalidCapsuleError(message, capsule.name)
    except Exception as e:
        message = f"This capsule is invalid for unknown reasons. Error: {e}"
        raise InvalidCapsuleError(message, capsule.name)


def capsule_module_name(data: bytes) -> str:
    """Creates a unique module name for the given capsule bytes"""
    hash_ = hashlib.sha256(data).hexdigest()
    return f"capsule_{hash_}"


_TYPE_CALLABLE = Union[type, callable]


def _validate_capsule_field(capsule: BaseCapsule,
                            name: str,
                            value: Any,
                            type_: _TYPE_CALLABLE) -> None:
    if type_ is callable:
        if not callable(value):
            raise InvalidCapsuleError(
                f"The capsule has an invalid internal configuration!\n"
                f"Capsule field {name} must be callable, got {type(value)}",
                capsule.name)
    elif not isinstance(value, type_):
        raise InvalidCapsuleError(
            f"The capsule has an invalid internal configuration!\n"
            f"Capsule field {name} must be of type {type_}, got {type(value)}",
            capsule.name)


def _validate_backend_field(capsule: BaseCapsule,
                            name: str,
                            value: Any,
                            type_: _TYPE_CALLABLE) -> None:
    if type_ is callable:
        if not callable(value):
            raise InvalidCapsuleError(
                f"The capsule's backend has an invalid configuration!\n"
                f"Backend field {name} must be callable, got {type(value)}",
                capsule.name)
    elif not isinstance(value, type_):
        raise InvalidCapsuleError(
            f"The capsule's backend has an invalid configuration!\n"
            f"Backend field {name} must be of type {type_}, got {type(value)}",
            capsule.name)
