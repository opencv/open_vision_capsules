from typing import Type, List, Any, Dict, Tuple, Any
from concurrent.futures import Future
import multiprocessing
import threading
import queue
import gc
from uuid import uuid4, UUID
from threading import RLock
from typing import NamedTuple

from vcap import BaseBackend, DETECTION_NODE_TYPE


class RpcRequest(NamedTuple):
    request_id: str
    """A string UUID"""
    function: str
    """The function to run"""
    args: Tuple[Any]
    """Arguments for the function"""
    kwargs: Dict[str, Any]
    """Keyword arguments for the function"""


class RpcResponse(NamedTuple):
    request_id: str
    """A string UUID corresponding to the RpcRequest"""
    result: Any
    """The result of the function that was run"""
    exception: BaseException
    """If the function threw an exception, this will hold the exception. 
    In the case when the function failed with an exceptio, RpcResponse.result
    will be None."""


def _rpc_server(
        incoming: multiprocessing.Queue,
        outgoing: multiprocessing.Queue,
        shutdown: multiprocessing.Event,
        num_workers: int,
        backend_class, args, kwargs):
    """ An RPC server for running Backend functions.
    :param incoming: Receives RpcRequests through here
    :param outgoing: Returns RpcResponses here
    :param shutdown: When set, this signals that the server should shut down.
    :param num_workers: How many internal threads should exist for processing
    incoming RpcRequests.
    :param backend_class: The backend to initialize
    :param args: *args for the backend_class __init__
    :param kwargs: *kwargs for the backend_class __init__
    """
    # Initialize the backend and catch any errors during initialization
    try:
        backend = backend_class(*args, **kwargs)
        outgoing.put(None)
    except BaseException as e:
        outgoing.put(e)
        return

    pool = multiprocessing.pool.ThreadPool(processes=num_workers)

    def handle_request(request: RpcRequest):
        try:
            result = getattr(backend, request.function)(  # noqa: F821
                *request.args, **request.kwargs)
            exception = None
        except BaseException as e:
            result = None
            exception = e
        outgoing.put(RpcResponse(
            request_id=request.request_id,
            result=result,
            exception=exception
        ))

    while not shutdown.is_set() or incoming.qsize():
        try:
            request: RpcRequest = incoming.get(timeout=0.1)
        except queue.Empty:
            continue
        pool.apply_async(handle_request, args=(request,))
    pool.close()
    pool.join()
    del backend
    # Make sure any __del__'s gets called, potentially freeing up
    # sockets that the backend may have been connected to. For example,
    # the OpenVINO HDDL plugin connects to a socket, and releases it
    # upon garbage collection.
    gc.collect()


class BackendRpcProcess(BaseBackend):
    """Runs a backend inside of another process.
    This functionality is very experimental! It may not work for all cases
    or frameworks.
    """

    def __init__(self, backend_class: Type[BaseBackend], *args, **kwargs):
        """
        :param backend_class: The Backend type to initialize
        :param args: Arguments for the backend
        :param kwargs: Keyword arguments for the backend
        """
        self._backend_class = backend_class
        self._incoming = multiprocessing.Queue()
        self._outgoing = multiprocessing.Queue()
        self._shutdown = multiprocessing.Event()
        self._futures_lock = RLock()
        self._futures: Dict[str, Future] = {}
        """Keep track of ongoing requests in a dict of request_id: Future """

        # Spin up the server process
        self._process = multiprocessing.Process(
            target=_rpc_server,
            daemon=True,
            name="RpcServer",
            kwargs={
                "outgoing": self._incoming,
                "incoming": self._outgoing,
                "shutdown": self._shutdown,
                "backend_class": backend_class,
                "num_workers": 100,
                "args": args,
                "kwargs": kwargs
            })
        self._process.start()
        # Wait for success or failure of the Backend initialization
        exception = self._incoming.get()
        if exception:
            raise exception

        # Start the client thread
        self._rpc_thread = threading.Thread(
            target=self._rpc_client,
            name="RpcClientThread")
        self._rpc_thread.start()

    @property
    def workload(self) -> int:
        # TODO: Use RPC to request the underlying Backends
        #       'workload' implementation
        with self._futures_lock:
            return len(self._futures)

    def _rpc_client(self):
        """This is the dedicated thread for receiving and routing results
        from the remote backend. It receives incoming RpcResponses and
        routes results or exceptions to the relevant Future objects."""
        while not self._shutdown.is_set() or self._incoming.qsize():
            try:
                response: RpcResponse = self._incoming.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._futures_lock:
                future = self._futures.pop(response.request_id)
            if response.exception:
                future.set_exception(response.exception)
            else:
                future.set_result(response.result)

    def _rpc_call(self, function: str, *args, **kwargs):
        """Run a function on the remote backend.
        :param function: The function to run on the remote backend
        :param args: Arguments for the function
        :param kwargs: Keyword arguments for the function
        """
        request = RpcRequest(
            function=function,
            args=args,
            kwargs=kwargs,
            request_id=uuid4().hex
        )

        future = Future()
        with self._futures_lock:
            self._futures[request.request_id] = future
        self._outgoing.put(request)
        return future.result()

    def process_frame(self, *args, **kwargs) -> DETECTION_NODE_TYPE:
        return self._rpc_call("process_frame", *args, **kwargs)

    def distances(self, *args, **kwargs):
        return self._rpc_call("distances", *args, **kwargs)

    def batch_predict(self, *args, **kwargs):
        """This function is implemented on the backend running on the backend
        process. It shouldn't be called from the parent process."""
        raise NotImplementedError()

    def close(self) -> None:
        # Close the underlying backend
        self._rpc_call("close")
        # Close the server process and the client thread
        self._shutdown.set()
        self._process.join()
        self._rpc_thread.join()
