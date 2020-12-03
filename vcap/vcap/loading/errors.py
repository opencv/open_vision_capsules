class CapsuleLoadError(Exception):
    """Raised when there was an error loading a capsule."""

    def __init__(self, message, capsule_name=None):
        self.capsule_name = capsule_name
        if capsule_name is not None:
            message = f"Capsule {capsule_name}: {message}"

        super().__init__(message)


class IncompatibleCapsuleError(CapsuleLoadError):
    """This is thrown when a capsule is of an incompatible API version for this
    version of the library.
    """


class InvalidCapsuleError(CapsuleLoadError):
    """This is thrown when a capsule has the correct API compatibility version,
    but it still doesn't have the correct information on it.
    """
