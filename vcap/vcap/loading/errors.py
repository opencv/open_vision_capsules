class IncompatibleCapsuleError(Exception):
    """This is thrown when a capsule is of an incompatible API version for this
    version of the library.
    """


class InvalidCapsuleError(Exception):
    """This is thrown when a capsule has the correct API compatibility version,
    but it still doesn't have the correct information on it.
    """
