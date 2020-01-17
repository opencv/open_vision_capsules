from Cryptodome.Cipher import AES


def encrypt(password, source, dest):
    """Encrypt the given source to dest.

    :param password: The password to encrypt with
    :param source: A file-like object to read from
    :param dest: A file-like object to write results to
    """
    cipher = AES.new(password.encode('utf-8'), AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(source.read())

    [dest.write(x) for x in (cipher.nonce, tag, ciphertext)]


def decrypt(password, source):
    """Decrypt the given source into bytes.

    :param password: The password to decrypt with
    :param source: A file-like object to read from
    :return: The resulting bytes
    """
    nonce, tag, ciphertext = [source.read(x) for x in (16, 16, -1)]

    cipher = AES.new(password.encode("utf-8"), AES.MODE_EAX, nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def decrypt_file(relative_path, password):
    """Decrypt the file given by the path.

    :param relative_path: A Path object describing the file to decrypt
    :param password: The password to decrypt with
    """
    path = relative_path
    with open(path, "rb") as f:
        return decrypt(password, f)


def encrypt_file(in_path, out_path, password):
    """Encrypt the file as a new file.

    Note that in_path should NOT be the same file as out_path, because the files
    are never fully loaded into memory.

    :param in_path: A Path object, the location of the unencrypted file
    :param out_path: A Path object, the location to save the encrypted data
    """
    with open(in_path, "rb") as in_file, open(out_path, "wb") as out_file:
        encrypt(password, in_file, out_file)
