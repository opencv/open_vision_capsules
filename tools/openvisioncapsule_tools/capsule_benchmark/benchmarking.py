import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, NamedTuple, Optional

import numpy as np
from tqdm import tqdm
from vcap import BaseCapsule, NodeDescription
from vcap.testing.input_output_validation import make_detection_node

from capsules import CapsuleDir
from workers import CapsuleThreadPool


class BenchmarkSuite:
    class Result(NamedTuple):
        capsule_name: str
        num_workers: int
        num_samples: int
        fps: float

    def __init__(self, capsule_dir: Path, num_workers: Iterable[int],
                 image_func: Optional[Callable[[], np.ndarray]] = None):
        """
        :param capsule_dir: Directory containing unpackaged capsules to test
        :param num_workers: Iterable containing the different num_worker values
            to test
        :param image_func: Function that returns an image. Can return the same
            image over and over, or different images
        """

        self.capsule_dir = CapsuleDir(capsule_dir)
        self.num_workers = num_workers

        # Generate a random image to run the benchmark on.
        # Generating an image for each process_frame has a big overhead,
        # limiting speed to < 100 FPS.
        self.rng = np.random.RandomState(1337)
        self.image = self.rng.randint(0, 255, (1920, 1080, 3),
                                      dtype=np.uint8)
        self.image.flags.writeable = False

        self.capsule_dir.package_capsules()

    def test(self, num_samples: int) -> List[Result]:
        results: List[self.Result] = []

        total_tests = len(self.capsule_dir) * len(list(self.num_workers))
        with tqdm(total=total_tests) as progress_bar:
            for capsule in self.capsule_dir:
                for num_workers in self.num_workers:
                    worker_pool = CapsuleThreadPool(num_workers)

                    duration = self.perform_test(capsule, worker_pool,
                                                 num_samples)
                    result = self.Result(
                        capsule_name=capsule.name,
                        num_workers=num_workers,
                        num_samples=num_samples,
                        fps=num_samples / duration.total_seconds()
                    )
                    results.append(result)

                    progress_bar.update(1)
                    worker_pool.shutdown()

                capsule.close()

        return results

    def generate_input_kwargs(self, capsule):
        if capsule.input_type.size is NodeDescription.Size.NONE:
            input_node = None
        else:
            input_node = make_detection_node(self.image.shape, capsule.input_type)

            # Set node size to cover entire frame
            height, width, _ = self.image.shape
            input_node.coords = [[0, 0],
                                 [width, 0],
                                 [width, height],
                                 [0, height]]

        if capsule.input_type.size is NodeDescription.Size.ALL:
            input_node = [input_node]

        return {"frame": self.image,
                "detection_node": input_node,
                "options": capsule.default_options,
                "state": capsule.stream_state()}

    def perform_test(self, capsule: BaseCapsule,
                     worker_pool: CapsuleThreadPool, num_samples: int) \
            -> datetime.timedelta:

        # Warm things up, such as getting model on GPU if capsule uses it
        warmup_results = worker_pool.map(
            lambda kwargs: capsule.process_frame(**kwargs),
            [self.generate_input_kwargs(capsule) for _ in range(50)]
        )
        for _ in warmup_results:
            pass

        # Generate test args before starting the test, so that the
        # benchmark is purely just for the capsule
        test_inputs = [self.generate_input_kwargs(capsule)
                       for _ in range(num_samples)]

        # Begin the benchmark
        start_time = datetime.datetime.now()
        results = worker_pool.map(
            lambda kwargs: capsule.process_frame(**kwargs),
            test_inputs)

        for _ in results:
            pass

        end_time = datetime.datetime.now()
        duration = end_time - start_time

        return duration
