from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from .crypto_utils import encrypt

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
            encrypt(key, saved_zip_bytes, output)
