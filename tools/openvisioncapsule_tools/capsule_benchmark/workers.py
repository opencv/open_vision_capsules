from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep


class CapsuleThreadPool(ThreadPoolExecutor):

    def __init__(self, num_workers: int):
        super().__init__(max_workers=num_workers)

        self.num_workers = num_workers

        self._warm_workers()

    def _warm_workers(self):
        """Initialize the worker pool before starting the test"""

        def waste_time(_):
            sleep(0.25)

        for _ in self.map(waste_time, range(self.num_workers)):
            pass
