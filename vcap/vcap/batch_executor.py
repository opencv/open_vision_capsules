import queue
from concurrent.futures import Future
from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterable, List, NamedTuple, Optional

from vcap import deprecated


class _Request(NamedTuple):
    """Used by BatchExecutor to keep track of requests and their respective
    future objects.
    """

    future: Future
    """The Future object for the BatchExecutor to return the output"""

    input_data: Any
    """A unit of input data expected by the batch_fn."""


class BatchExecutor:
    """Feeds jobs into batch_fn in batches, returns results through Futures.

    This class simplifies centralizing work from a multitude of sources and
    running that work in a batched predict function, then returning that
    work to the respective Futures.
    """

    def __init__(self,
                 batch_fn: Callable[[List[Any]], Iterable[Any]],
                 max_batch_size=40,
                 num_workers: int = 1):
        """Initialize a new BatchExecutor

        :param batch_fn: A function that takes in a list of inputs and iterates
        the outputs in the same order as the inputs.
        :param max_batch_size: The maximum length of list to feed to batch_fn
        :param num_workers: How many workers should be calling batch_fn
        """
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self._request_queue: Queue[_Request] = Queue()
        self.workers = [Thread(target=self._worker,
                               daemon=True,
                               name="BatchExecutorThread")
                        for _ in range(num_workers)]

        # The number of images currently in the work queue or being processed
        self._num_imgs_being_processed: int = 0

        self._running: bool = True

        for worker in self.workers:
            worker.start()

    @property
    def total_imgs_in_pipeline(self) -> int:
        return self._request_queue.qsize() + self._num_imgs_being_processed

    def submit(self, input_data: Any, future: Future = None) -> Future:
        """Submits a job and returns a Future that will be fulfilled later."""
        future = future or Future()

        self._request_queue.put(_Request(
            future=future,
            input_data=input_data))
        return future

    def _on_requests_ready(self, batch: List[_Request]) -> None:
        """Push inputs through the given prediction backend

        :param batch: A list of requests to work on
        """
        # Extract the futures from the requests
        inputs: List[Any] = [req.input_data for req in batch]
        futures: List[Future] = [req.future for req in batch]

        # Route the results to each request
        try:
            for prediction in self.batch_fn(inputs):
                # Popping the futures ensures that if an error occurs, only
                # the futures that haven't had a result set will have
                # set_exception called
                futures.pop(0).set_result(prediction)
        except BaseException as exc:
            # Catch exceptions and pass them to the futures, similar to the
            # ThreadPoolExecutor implementation:
            # https://github.com/python/cpython/blob/91e93794/Lib/concurrent/futures/thread.py#L51
            for future in futures:
                future.set_exception(exc)

    def _worker(self):
        self._running = True

        while self._running:
            # Get a new batch
            batch = self._get_next_batch()

            # If no batch was able to be retrieved, restart the loop
            if batch is None:
                continue

            # Check to make sure the thread isn't trying to end
            if not self._running:
                break

            self._num_imgs_being_processed += len(batch)
            self._on_requests_ready(batch)
            self._num_imgs_being_processed -= len(batch)

        self._running = False

    def _get_next_batch(self) -> Optional[List[_Request]]:
        """A helper function to help make the main thread loop more readable
        :returns: A non-empty list of collected items, or None if the worker is
                  no longer running (i.e. self._continue == False)
        """
        batch: List[_Request] = []
        while len(batch) < self.max_batch_size:
            # Check if there's a new request
            try:
                # Try to get a new request. Have a timeout to check if closing
                new_request = self._request_queue.get(timeout=.1)
            except queue.Empty:
                # If the thread is being requested to close, exit early
                if not self._running:
                    return None

                # Wait for requests again
                continue

            batch.append(new_request)

            # If the request queue is now empty, let worker run everything in
            # the batch
            if self._request_queue.empty():
                break

        return batch

    def close(self) -> None:
        """Stop the BatchExecutor gracefully."""
        self._running = False
        for worker in self.workers:
            worker.join()
