from typing import Type, List, Any, Dict, Tuple, Any
import multiprocessing
import threading
import queue
from uuid import uuid4, UUID
from threading import RLock
from typing import NamedTuple
from socketserver import BaseRequestHandler

import numpy as np
from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState


class RpcRequest(NamedTuple):
    function: str
    args: Tuple[Any]
    kwargs: Dict[str, Any]
    request_id: UUID


def _server(incoming,
            outgoing,
            shutdown: multiprocessing.Event,
            backend_class, args, kwargs):
    """The other process server."""
    backend = backend_class(*args, **kwargs)
    outgoing.put(None)

    while not shutdown.is_set() or incoming.qsize():
        try:
            request: RpcRequest = incoming.get(timeout=0.1)
        except queue.Empty:
            continue
        result = getattr(backend, request.function)(
            *request.args, **request.kwargs)
        outgoing.put((request.request_id, result))


class MockOven:
    """I will figure this out... later."""
    total_imgs_in_pipeline = 0

    def close(self):
        pass


class BackendProcess(BaseBackend):
    """Runs a backend inside of another process.
    This functionality is very experimental! It may not work for all cases
    or frameworks.
    """

    def __init__(self, backend_class: Type[BaseBackend], *args, **kwargs):
        self._backend_class = backend_class
        self._incoming = multiprocessing.Queue()
        self._outgoing = multiprocessing.Queue()
        """The syntax is to send the function name, args, kwargs, and 
        a return queue for the response."""
        self._shutdown = multiprocessing.Event()
        self._process = multiprocessing.Process(
            target=_server,
            kwargs={
                "outgoing": self._incoming,
                "incoming": self._outgoing,
                "shutdown": self._shutdown,
                "backend_class": backend_class,
                "args": args,
                "kwargs": kwargs
            })
        self._rpc_thread = threading.Thread(target=self._rpc_client)

        # Start the process and wait for startup to finish
        self._process.start()
        self._incoming.get()

        # Start the consumer thread
        self._rpc_thread.start()
        print("Startup Finished!")

        # Keep track of result queues in a dict of request_id: result_queue
        self._result_queues: Dict[UUID, queue.Queue] = {}
        self.oven = MockOven()

    def _rpc_client(self):
        while not self._shutdown.is_set():
            try:
                result_id, result = self._incoming.get(timeout=0.1)
            except queue.Empty:
                continue
            result_queue = self._result_queues.pop(result_id)
            result_queue.put(result)

    def _rpc_call(self, function, *args, **kwargs):
        print("RPC Calling", function)
        request = RpcRequest(
            function=function,
            args=args,
            kwargs=kwargs,
            request_id=uuid4()
        )
        result_queue = queue.Queue()
        self._result_queues[request.request_id] = result_queue
        self._outgoing.put(request)
        return result_queue.get()

    def process_frame(self, *args, **kwargs) -> DETECTION_NODE_TYPE:
        return self._rpc_call("process_frame", *args, **kwargs)

    def batch_predict(self, *args, **kwargs):
        """This function is implemented on the backend running on the backend
        process. It shouldn't be called from the parent process."""
        raise NotImplementedError()

    def close(self) -> None:
        self._rpc_call("close")

        self._shutdown.set()
        self._process.join()
        self._rpc_thread.join()
