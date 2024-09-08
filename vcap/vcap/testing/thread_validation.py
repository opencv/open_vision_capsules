import re
import threading
from typing import List, Optional


def verify_all_threads_closed(allowable_threads: Optional[List[str]] = None):
    """A convenient function to throw an error if all threads are not closed.

    :param allowable_threads: A list of regular expressions that match to
        thread names that are allowed to stay open
    """
    if allowable_threads is None:
        allowable_threads = []
    allowable_threads += [
        r"pydevd\.Writer",
        r"pydevd\.Reader",
        r"pydevd\.CommandThread",
        r"profiler\.Reader",
        r"MainThread",
        r"ThreadPoolExecutor-\d+_\d+"
    ]

    open_threads = []

    for thread in threading.enumerate():
        matched = False
        for name_pattern in allowable_threads:
            if re.match(name_pattern, thread.name):
                matched = True
                break

        if not matched:
            open_threads.append(thread)

    if len(open_threads) != 0:
        raise EnvironmentError(
            "Not all threads were shut down! Currently running threads: " +
            str(open_threads))
