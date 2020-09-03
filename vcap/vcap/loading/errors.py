class CapsuleError(Exception):
    """Raised when there was an error loading a capsule."""

    def __init__(self, capsule_name, message):
        self.capsule_name = capsule_name
        super().__init__(f"Capsule {capsule_name}: {message}")


class IncompatibleCapsuleError(CapsuleError):
    """This is thrown when a capsule is of an incompatible API version for this
    version of the library.
    """


class InvalidCapsuleError(CapsuleError):
    """This is thrown when a capsule has the correct API compatibility version,
    but it still doesn't have the correct information on it.
    """
