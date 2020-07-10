import configparser
import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Callable, List, Optional, Union
from zipfile import ZipFile

from vcap import BaseCapsule, BaseBackend, BaseStreamState, NodeDescription
from vcap.loading.errors import IncompatibleCapsuleError, InvalidCapsuleError

from .crypto_utils import decrypt_file
from .import_hacks import ZipFinder
from .packaging import CAPSULE_FILE_NAME, META_FILE_NAME

MAJOR_COMPATIBLE_VERSION = 0
MINOR_COMPATIBLE_VERSION = 2
"""The capsule version that this version of the library supports."""

MAJOR_MINOR_SEMVER_PATTERN = re.compile(r"([0-9]+)\.([0-9]+)")


def load_capsule(path: Union[str, Path],
                 key=None, inference_mode=True) -> BaseCapsule:
    """Load a capsule from the filesystem.

    :param path: The path to the capsule file
    :param key: The AES key to decrypt the capsule with, or None if the capsule
        is not encrypted
    :param inference_mode: If True, the backends for this capsule will be
        started. If False, the capsule will never be able to run inference, but
        it will still have it's various readable attributes.
    """
    path = Path(path)
    loaded_files = {}

    if key is None:
        # Capsule is unencrypted and already a zip file
        capsule_data = path.read_bytes()
    else:
        # Decrypt the capsule into its original form, a zip file
        capsule_data = decrypt_file(path, key)
    file_like = BytesIO(capsule_data)

    code = None
    with ZipFile(file_like, "r") as capsule_file:
        if CAPSULE_FILE_NAME not in capsule_file.namelist():
            raise RuntimeError(f"Capsule {path} has no {CAPSULE_FILE_NAME}")

        if META_FILE_NAME not in capsule_file.namelist():
            raise IncompatibleCapsuleError(
                f"Capsule {path} has no {META_FILE_NAME}")

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
                f"Capsule {path} is not compatible with this software. The "
                f"capsule's OpenVisionCapsules required major version is "
                f"{major} but this software uses OpenVisionCapsules "
                f"{MAJOR_COMPATIBLE_VERSION}.{MINOR_COMPATIBLE_VERSION}.")
        if minor > MINOR_COMPATIBLE_VERSION:
            raise IncompatibleCapsuleError(
                f"Capsule {path} requires a version of OpenVisionCapsules "
                f"that is too new for this software. The capsule requires at "
                f"least version {major}.{minor} but this software uses "
                f"OpenVisionCapsules "
                f"{MAJOR_COMPATIBLE_VERSION}.{MINOR_COMPATIBLE_VERSION}.")

        # With the capsule's code loaded, initialize the object
        capsule_module = ModuleType(path.stem)
        try:
            # Allow the capsule.py to import other files in the capsule
            capsule_dir_path = (path.parent / path.stem).absolute()
            sys.meta_path.insert(1, ZipFinder(capsule_file, capsule_dir_path))

            # Run the capsule
            compiled = compile(code, capsule_dir_path / "capsule.py", "exec")
            exec(compiled, capsule_module.__dict__)
        except Exception as e:
            raise InvalidCapsuleError(
                "Could not execute the code in the capsule!\n"
                f"File: {path}\n"
                f"Error: {e}")
        finally:
            # Remove custom import code
            sys.meta_path.pop(1)

    # noinspection PyUnresolvedReferences
    new_capsule: BaseCapsule = capsule_module.Capsule(
        capsule_files=loaded_files,
        inference_mode=inference_mode)

    try:
        _validate_capsule(new_capsule)
    except InvalidCapsuleError as e:
        logging.warning(f"Failed to load capsule {path}")
        new_capsule.close()
        raise e

    return new_capsule


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
                "underscores")

        # Validate the capsule class attributes
        capsule_assertions = [
            isinstance(capsule.name, str),
            callable(capsule.backend_loader),
            isinstance(capsule.version, int),
            isinstance(capsule.input_type, NodeDescription),
            isinstance(capsule.output_type, NodeDescription),
            isinstance(capsule.options, dict)]

        if not all(capsule_assertions):
            raise InvalidCapsuleError(
                f"The capsule has an invalid internal configuration!\n" +
                f"Capsule Assertions: {capsule_assertions}")

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
                    f"{unwanted_attr}")
            except AttributeError:
                pass

        # Validate the capsule's backend_loader takes the right args
        loader = capsule.backend_loader
        loader_assertions = [
            callable(loader),
            check_arg_names(func=loader, correct=["capsule_files", "device"])]
        if not all(loader_assertions):
            raise InvalidCapsuleError(
                f"The capsule's backend_loader has an invalid configuration!\n"
                f"Loader Assertions: {loader_assertions}")

        # Validate the backend class attributes
        if capsule.backends is not None:
            backend = capsule.backends[0]
            backend_assertions = [
                callable(backend.batch_predict),
                callable(backend.process_frame),
                callable(backend.close),
                isinstance(capsule.backends[0], BaseBackend)]
            if not all(backend_assertions):
                raise InvalidCapsuleError(
                    f"The capsule's backend has an invalid configuration!\n"
                    f"Backend Assertions: {backend_assertions}")

        # Validate the stream state
        stream_state = capsule.stream_state
        stream_state_assertions = [
            (stream_state is BaseStreamState or
             BaseStreamState in stream_state.__bases__)]
        if not all(stream_state_assertions):
            raise InvalidCapsuleError(
                "The capsule's stream_state has an invalid configuration!\n"
                f"Stream State Assertions: {stream_state_assertions}")

        # Validate that if the capsule is an encoder, it has a threshold option
        if capsule.capability.encoded:
            if "recognition_threshold" not in capsule.options.keys():
                raise InvalidCapsuleError(
                    "This capsule can encode, but doesn't have a option "
                    "called \"recognition_threshold\"!")
    except InvalidCapsuleError:
        # Don't catch this exception type in the "except Exception" below
        raise
    except AttributeError as e:
        message = f"The capsule did not describe the necessary attributes. " \
                  f"Error: {e}"
        raise InvalidCapsuleError(message)
    except Exception as e:
        message = f"This capsule is invalid for unknown reasons. Error: {e}"
        raise InvalidCapsuleError(message)
