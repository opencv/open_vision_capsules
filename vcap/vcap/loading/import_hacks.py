"""Code which changes the way Python processes imports. Allows for capsules to
import Python modules and packages within the capsule itself.
"""
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
import os
from pathlib import Path
from zipfile import ZipFile


class ZipFinder(MetaPathFinder):
    """An import finder that allows modules and packages inside a zip file to be
    imported.
    """

    def __init__(self, zip_file: ZipFile,
                 capsule_dir_path: Path,
                 root_package_name: str):
        """
        :param zip_file: The ZipFile loaded in memory
        :param capsule_dir_path:
            The path to the directory where the development version of the
            capsule is stored. For example, if the capsule being loaded is in
            ../capsules/my_capsule_name.py

            Then capsule_dir_path would be
            ../capsules/my_capsule_name/

            The reason for this is so that debugging can work within capsules.
            When a capsule is executed, it is "compiled" to this path, so that
            debug information still works (along with breakpoints!)
        :param root_package_name: The name of the root package that this zip
            file provides
        """
        self._zip_file = zip_file
        self._capsule_dir_path = capsule_dir_path
        self._capsule_dir_name = self._capsule_dir_path.name
        self._root_package_name = root_package_name

    def find_spec(self, fullname, _path=None, _target=None):
        if not self._in_capsule(fullname):
            # If the capsule directory name is not in the fullname, it's
            # not our job to import it
            return None

        if fullname == self._root_package_name:
            # If the capsule root directory is being loaded, return a
            # modulespec
            return ModuleSpec(
                name=fullname,
                loader=None,
                is_package=True)

        # Get rid of the capsule name prefix
        pruned_fullname = _remove_capsule_name(self._capsule_dir_name,
                                               fullname)
        package_path = _package_fullname_to_path(pruned_fullname)
        module_path = _module_fullname_to_path(pruned_fullname)

        if package_path in self._zip_file.namelist():
            # If a directory exists with a name that matches the import, we
            # assume it is a package import.
            return ModuleSpec(name=fullname,
                              loader=None,
                              is_package=True)
        elif module_path in self._zip_file.namelist():
            # If a .py file exists with a name that matches the import, we
            # assume it is a module import
            module_file_path = self._capsule_dir_path / module_path
            loader = ZipModuleLoader(zip_file=self._zip_file,
                                     module_file_path=module_file_path,
                                     capsule_dir_name=self._capsule_dir_name)
            return ModuleSpec(name=fullname,
                              loader=loader)

        raise ImportError(f"Problem while importing {fullname}")

    def _in_capsule(self, fullname):
        parts = fullname.split(".")
        if parts[0] != self._root_package_name:
            return False
        return True


class ZipModuleLoader(Loader):
    """Loads modules from a zip file."""

    def __init__(self, zip_file: ZipFile,
                 module_file_path: Path,
                 capsule_dir_name: str):
        """
        :param zip_file: The ZipFile loaded in memory
        :param module_file_path: The path to where the python file would be
            in the filesystem if it was being run directly as opposed to in a
            *.cap zip.
        """
        self._zip_file = zip_file
        self._module_file_path = module_file_path
        self._capsule_dir_name = capsule_dir_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pruned_name = _remove_capsule_name(
            capsule_name=self._capsule_dir_name,
            fullname=module.__name__)
        zip_path = _module_fullname_to_path(pruned_name)

        # Extract the code from the zip
        code = self._zip_file.read(zip_path)

        # Compile code with the file path set first, so that debugging and
        # tracebacks work in development (they reference a file)
        # also breakpoints work in IDE's, which is quite helpful.
        compiled = compile(code, self._module_file_path, "exec")
        exec(compiled, module.__dict__)
        return module


def _package_fullname_to_path(fullname):
    """Converts a package's fullname to a file path that should be the package's
    directory.

    :param fullname: The fullname of a package, like package_a.package_b
    :return: A derived filepath, like package_a/package_b
    """
    return fullname.replace(".", os.sep) + os.sep


def _module_fullname_to_path(fullname):
    """Converts a module's fullname to a file path that should be the module's
    Python file.

    :param fullname: The fullname of a module, like package_a.my_module
    :return: A derived filepath, like package_a/my_module.py
    """
    return fullname.replace(".", os.sep) + ".py"


def _remove_capsule_name(capsule_name, fullname):
    """Remove "capsule_name" from capsule_name.some_module.some_module2
    Since the files in the zip file won't have "capsule_name" in the paths
    """
    parts = fullname.split(".")
    return ".".join(parts[1:])
