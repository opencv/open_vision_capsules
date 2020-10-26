from concurrent.futures import Future
from typing import List, Generator, Any, Tuple
import random

import pytest

from vcap.ovens import Oven, _OvenRequest


@pytest.fixture()
def oven():
    """To use this fixture, replace oven.batch_fn with your own
    batch function."""

    def batch_fn(inputs):
        raise NotImplemented

    oven = Oven(batch_fn=batch_fn)
    yield oven
    oven.close()


def batch_fn_base(inputs: List[int], raises: bool) \
        -> Generator[Any, None, None]:
    """Process results and yield them as they are processed
    :param inputs: A list of inputs
    :param raises: If True, raises an error on the 5th input
    """
    for i in inputs:
        if i == 5 and raises:
            raise RuntimeError("Oh no, a batch_fn error has occurred!")
        yield i * 100


def batch_fn_returns_generator(inputs: List[int]) \
        -> Generator[Any, None, None]:
    return (o for o in batch_fn_base(inputs, raises=False))


def batch_fn_returns_generator_raises(inputs: List[int]) \
        -> Generator[Any, None, None]:
    return (o for o in batch_fn_base(inputs, raises=True))


def batch_fn_returns_list(inputs: List[int]) -> List[Any]:
    """Process results and yield them at the end, as a list."""
    return list(batch_fn_base(inputs, raises=False))


def batch_fn_returns_list_raises(inputs: List[int]) -> List[Any]:
    return list(batch_fn_base(inputs, raises=True))


@pytest.mark.parametrize(
    argnames=["batch_fn", "expect_partial_results"],
    argvalues=[
        (batch_fn_returns_generator_raises, True),
        (batch_fn_returns_list_raises, False)
    ]
)
def test_exceptions_during_batch_fn(oven, batch_fn, expect_partial_results):
    """Test that ovens catch exceptions that occur in the batch_fn and
     propagate them through the requests Future objects.

     If an exception occurs after processing some of the batch, the expectation
     is that the unprocessed inputs of the batch will get an exception
     set (expect_partial_results=True). If the exception happens before
     receiving any results, all future objects should have exceptions set.
     """
    oven.batch_fn = batch_fn
    request_batch = [
        _OvenRequest(
            future=Future(),
            input_data=i)
        for i in range(10)
    ]
    oven._on_requests_ready(request_batch)
    for i, request in enumerate(request_batch):
        print("Running req", i, request)
        if expect_partial_results and i < 5:
            result = request.future.result(timeout=5)
            assert result == request.input_data * 100, \
                "The result for this future doesn't match the input that " \
                "was supposed to have been routed to it!"
        else:
            with pytest.raises(RuntimeError):
                request.future.result(timeout=5)


@pytest.mark.parametrize(
    argnames=["batch_fn"],
    argvalues=[
        (batch_fn_returns_generator,),
        (batch_fn_returns_list,)
    ]
)
def test_relevant_input_outputs_match(oven, batch_fn):
    """Test the output for any given input is routed to the correct
    Future object. """
    oven.batch_fn = batch_fn

    # Submit input values in a random order
    request_inputs = list(range(10000))
    random.seed("vcap? More like vgood")
    random.shuffle(request_inputs)

    # Submit results to the oven and mark the input for any particular future
    inputs_and_futures: List[Tuple[int, Future]] = []
    for input_data in request_inputs:
        future = oven.submit(input_data)
        inputs_and_futures.append((input_data, future))

    # Verify that all outputs are the expected ones for their respective input
    for input_data, future in inputs_and_futures:
        result = future.result(timeout=5)
        assert result == input_data * 100, \
            "The result for this future doesn't match the input that " \
            "was supposed to have been routed to it!"

    assert oven.total_imgs_in_pipeline == 0
