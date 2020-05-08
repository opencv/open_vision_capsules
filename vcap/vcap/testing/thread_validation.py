import threading


def verify_all_threads_closed(allowable_threads=None):
    """A convenient function to throw an error if all threads are not closed"""
    if allowable_threads is None:
        allowable_threads = []
    allowable_threads += ['pydevd.Writer',
                          'pydevd.Reader',
                          'pydevd.CommandThread',
                          'profiler.Reader',
                          'MainThread']

    open_threads = [t.name for t in threading.enumerate()
                    if t.name not in allowable_threads]

    if len(open_threads) != 0:
        raise EnvironmentError(
            "Not all threads were shut down! Currently running threads: " +
            str(open_threads))
