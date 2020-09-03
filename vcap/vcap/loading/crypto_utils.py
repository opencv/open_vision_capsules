from typing import BinaryIO

from Cryptodome.Cipher import AES


def encrypt(password: str, data: bytes, dest: BinaryIO) -> None:
    """Encrypt the given source to dest.

    :param password: The password to encrypt with
    :param data: Data to encrypt
    :param dest: A file-like object to write results to
    """
    cipher = AES.new(password.encode('utf-8'), AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)

    [dest.write(x) for x in (cipher.nonce, tag, ciphertext)]


def decrypt(password: str, data: bytes) -> bytes:
    """Decrypt the given bytes.

    :param password: The password to decrypt with
    :param data: Data to decrypt
    :return: The resulting bytes
    """
    nonce = data[:16]
    tag = data[16:32]
    ciphertext = data[32:]

    cipher = AES.new(password.encode("utf-8"), AES.MODE_EAX, nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)
