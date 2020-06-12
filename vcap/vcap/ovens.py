"""Defines ovens.

Ovens are classes that know how to do some kind of generic work in batches.
"""
import logging
import queue
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, NamedTuple, Optional


class _OvenRequest(NamedTuple):
    """A request that is sent to an oven to do some work on an image, and
    push predictions into the output_queue

    output_queue: The queue for the oven to put the results in
    img_bgr: An OpenCV BGR image to run detection on
    """
    output_queue: Queue
    input_data: Any


class Oven:
    """This class simplifies receiving work from a multitude of sources and
    running that work in a batched predict function, then returning that
    work to the respective output queues."""

    def __init__(self, batch_fn: Callable[[List[Any]], List[Any]],
                 max_batch_size=40,
                 num_workers: int = 1):
        """Initialize a new oven.
         that the oven will wait between running a batch regardless of batch
         size.

        :param batch_fn: A function that takes in a list of inputs and iterates
        the outputs in the same order as the inputs.
        :param max_batch_size: The maximum length of list to feed to batch_fn
        :param num_workers: How many workers should be calling batch_fn
        """
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self._request_queue = Queue()
        self.workers = [Thread(target=self._worker,
                               daemon=True,
                               name="OvenThread")
                        for _ in range(num_workers)]

        # The number of images currently in the work queue or being processed
        self._num_imgs_being_processed: int = 0

        self._running: bool = True

        for worker in self.workers:
            worker.start()

    @property
    def total_imgs_in_pipeline(self) -> int:
        return self._request_queue.qsize() + self._num_imgs_being_processed

    def submit(self, input_data: Any, output_queue: Queue = None) -> Queue:
        """Creates an OvenRequest for you and returns the output queue"""
        output_queue = output_queue if output_queue else Queue()
        self._request_queue.put(_OvenRequest(
            output_queue=output_queue,
            input_data=input_data))
        return output_queue

    def _on_requests_ready(self, batch: List[_OvenRequest]) -> None:
        """Push images through the given prediction backend

        :param batch: A list of requests to work on
        """
        # Extract the images from the requests
        inputs: List[Any] = [req.input_data for req in batch]
        output_queues: List[Queue] = [req.output_queue for req in batch]

        # Route the results to each request
        predictions: List[Any] = self.batch_fn(inputs)

        response_count = 0
        for response_count, output in enumerate(predictions, start=1):
            output_queues[response_count - 1].put(output)

        if response_count != len(inputs):
            logging.error(f"CRITICAL ERROR: Backend returned {response_count} "
                          f"responses. Expected {len(inputs)}")

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

    def _get_next_batch(self) -> Optional[List[_OvenRequest]]:
        """A helper function to help make the main thread loop more readable
        :returns: A non-empty list of collected items, or None if the worker is
                  no longer running (i.e. self._continue == False)
        """
        batch: List[_OvenRequest] = []
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
        """Stop the oven gracefully."""
        self._running = False
        for worker in self.workers:
            worker.join()
