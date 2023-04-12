#
# Copyright (c) 2021 Dilili Labs, Inc.  All rights reserved. Contains Dilili Labs Proprietary Information. RESTRICTED COMPUTER SOFTWARE.  LIMITED RIGHTS DATA.
#
# This script compiles a Python app using Cython.

import multiprocessing
import re
import subprocess
import sysconfig
from distutils.cmd import Command
from distutils.command.build_ext import build_ext
from pathlib import Path

from Cython.Build import cythonize
from Cython.Compiler import Options as CompilerOptions
from setuptools import setup
from setuptools.extension import Extension

THREAD_COUNT = multiprocessing.cpu_count() * 2

BUILD_PATH = Path("build/")
PYTHON_VER = "python3.8"

BFAPP_PY_DIRS = [
    "capsule_infer",
    "capsule_classifier_accuracy",
]

# Used to exclude various Python files from Cythonization. If the path is a
# file, then that file will not by Cythonized. If the path is a directory, then
# everything in that directory will not be Cythonized.
cythonize_excludes = [
    # These aren't runtime files
    r"tests/.*",
]


def is_excluded(path):
    """
    :param path: The path to check
    :return: True if the file should be excluded from build-related actions
    """
    for exclude in cythonize_excludes:
        if re.match(exclude, str(path)):
            return True

    return False


class BuildBinariesCommand(Command):
    """Builds a Python App run files into a binary."""

    description = "build the main scripts into binaries"
    user_options = []

    def __init__(self, dist, executable_paths):
        self._executable_paths = executable_paths
        super().__init__(dist)

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path in self._executable_paths:
            self._build_binary(path)

    @staticmethod
    def _build_binary(source: Path):
        c_file = source.parent / (source.stem + ".c")
        exe_file = source.with_suffix("")

        cython_command = f"python3 -m cython --embed -3 -o {c_file} {source}"
        subprocess.run(cython_command, shell=True, check=True)

        gcc_command = (
            f"gcc -Os "
            f"-I {sysconfig.get_path('include')} "
            f"-o {exe_file} "
            f"{c_file} "
            f"-l{PYTHON_VER} -lpthread -lm -lutil -ldl "
        )
        subprocess.run(gcc_command, shell=True, check=True)


class BuildBinariesCommand(BuildBinariesCommand):
    def __init__(self, dist):
        binaries = []
        for BFAPP_PY_DIR in BFAPP_PY_DIRS:
            binaries += list(Path(BFAPP_PY_DIR).glob(BFAPP_PY_DIR + ".py"))
        super().__init__(dist, binaries)


class CleanCommand(Command):
    """Cleans the resulting build files from the project directory."""

    description = "delete output files from the build"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        clean_paths = (Path(BFAPP_PY_DIR) for BFAPP_PY_DIR in BFAPP_PY_DIRS)
        clean_extensions = ["*.so", "*.c", "*.html"] + BFAPP_PY_DIRS

        for clean_path in clean_paths:
            for clean_extension in clean_extensions:
                for clean_file in clean_path.rglob(clean_extension):
                    if not is_excluded(clean_file):
                        print(f"delete {clean_file}")
                        clean_file.unlink()

        # Delete the output files from creating the executables
        build_folder_files = list(BUILD_PATH.glob("**/**/*"))

        for path in build_folder_files:  # executable_paths:
            if path.is_file():
                print(f"delete {path}")
                path.unlink()


class BuildExtParallel(build_ext):
    def __init__(self, dist, extension_path):
        self._extension_path = extension_path

        dist.ext_modules = cythonize(
            self._find_extension_modules(),
            compiler_directives={
                "language_level": "3",
            },
            annotate=False,
            nthreads=THREAD_COUNT,
        )
        super().__init__(dist)

    def finalize_options(self):
        super().finalize_options()
        if self.parallel is None:
            self.parallel = THREAD_COUNT

    def _find_extension_modules(self):
        extension_modules = []
        for extension_path in self._extension_path:
            for py_file in extension_path.rglob("*.py"):
                if not is_excluded(py_file):
                    mod_path = str(py_file).replace("/", ".").replace(".py", "")
                    extension = Extension(
                        # In past releases, -O3 causes segfaults after ~4-6 hours of use.
                        # If changed, make sure to run 24h to confirm no stability regressions.
                        mod_path,
                        [str(py_file)],
                        extra_compile_args=["-O1"],
                    )
                    extension_modules.append(extension)
        return extension_modules


class BuildExtParallel(BuildExtParallel):
    def __init__(self, dist):
        paths = []
        paths += (Path(BFAPP_PY_DIR) for BFAPP_PY_DIR in BFAPP_PY_DIRS)
        super().__init__(dist, paths)


"""Information on Compiler Options:
https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
"""
CompilerOptions.error_on_unknown_names = True
CompilerOptions.error_on_uninitialized = True

setup(
    name="open_vision_capsule_tools_cython",
    cmdclass={
        "build_ext": BuildExtParallel,
        "build_binaries": BuildBinariesCommand,
        "clean": CleanCommand,
    },
)
