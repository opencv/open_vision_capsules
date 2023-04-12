from argparse import ArgumentParser

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from typing import Optional, Union

from vcap.loading.crypto_utils import encrypt, decrypt
import os

CAPSULE_EXTENSION = ".cap"
"""The file extension that capsule files use."""

CAPSULE_FILE_NAME = "capsule.py"
"""The name of the main file in the capsule."""

META_FILE_NAME = "meta.conf"
"""The name of the meta file that contains information such as the API 
compatibility version of this capsule"""


class UnpackagedCapsuleError(Exception):
    pass


def package_capsule(unpackaged_dir: Path, output_file: Path, key=None):
    """Packages an unpackaged capsule up into a zip and applies encryption.

    :param unpackaged_dir: The directory to package up
    :param output_file: The name and location to save the packaged capsule
    :param key: An AES key to encrypt the capsule with
    """
    # Check to make sure there is a capsule.py
    if len(list(unpackaged_dir.glob(CAPSULE_FILE_NAME))) == 0:
        raise UnpackagedCapsuleError(
            f"Unpackaged capsule {unpackaged_dir} is missing a "
            f"{CAPSULE_FILE_NAME}")
    # Check to make sure there's a meta.conf
    if len(list(unpackaged_dir.glob(META_FILE_NAME))) == 0:
        raise UnpackagedCapsuleError(
            f"Unpackaged capsule {unpackaged_dir} is missing a "
            f"{META_FILE_NAME}")

    saved_zip_bytes = BytesIO()
    with ZipFile(saved_zip_bytes, "w") as capsule_file:
        for f in unpackaged_dir.glob("**/*"):
            relative_name = f.relative_to(unpackaged_dir)
            capsule_file.write(f, relative_name)

    with output_file.open("wb") as output:
        # Move the read cursor to the start of the zip data
        saved_zip_bytes.seek(0)
        if key is None:
            output.write(saved_zip_bytes.read())
        else:
            encrypt(key, saved_zip_bytes.read(), output)


def unpackage_capsule(capsule_file: Union[str, Path],
        unpackage_to: Union[str, Path],
        key: Optional[str] = None):
    """Unpackagea capsule from the given bytes.

    :param path: The path to the capsule file
    :param key: The AES key to decrypt the capsule with, or None if the capsule
        is not encrypted
    :return: None
    """
    path = Path(capsule_file)

    data=path.read_bytes()

    if key is not None:
        # Decrypt the capsule into its original form, a zip file
        data = decrypt(key, data)

    file_like = BytesIO(data)

    with ZipFile(file_like, "r") as capsule_file:
        os.makedirs(unpackage_to, exist_ok=True)
        for name in capsule_file.namelist():

            output_path = Path(unpackage_to + '/' + name)
            print(output_path)

            if name.endswith('/'):
                os.makedirs(output_path, exist_ok=True)
            else:
                output_file = capsule_file.read(name)
                with output_path.open("wb") as output:
                    output.write(output_file)


def packaging_parse_args():
    parser = ArgumentParser(description='Package capsules and unpackage a capsule file')
    parser.add_argument(
        "--capsule-dir",
        type=Path,
        required=False,
        help="The parent directory with capsule directories for packaging"
    )
    parser.add_argument(
        "--capsule-file",
        type=Path,
        required=False,
        help="The capsule for unpackaging"
    )
    parser.add_argument(
        "--capsule-key",
        type=str,
        required=False,
        help="The encrption key for packaging or unpackaging"
    )
    return parser


def packaging(capsule_dir, capsule_file, capsule_key):
    if capsule_dir is not None:
        for path in capsule_dir.iterdir():
            if path.is_dir():
                package_capsule(
                    unpackaged_dir=path,
                    output_file=path.with_suffix(CAPSULE_EXTENSION),
                    key=capsule_key,
                )
    elif capsule_file is not None:
        capsule_filepath, file_ext = os.path.splitext(capsule_file)
        if file_ext == CAPSULE_EXTENSION:
            unpackage_capsule(
                capsule_file = capsule_file,
                unpackage_to = capsule_filepath,
                key = capsule_key
            )
        else:
            return False
    else:
        return False

    return True


def packaging_main():
    parser = packaging_parse_args()
    args = parser.parse_args()
    rtn = packaging(args.capsule_dir, args.capsule_file, args.capsule_key)
    if rtn == False:
        parser.print_help()


if __name__ == "__main__":
    packaging_main()
