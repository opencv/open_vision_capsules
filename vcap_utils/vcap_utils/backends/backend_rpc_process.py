import multiprocessing
import threading
import queue
import gc
from typing import List, Dict, Tuple, Any, NoReturn
from concurrent.futures import Future
from uuid import uuid4
from typing import NamedTuple, Type

import numpy as np

from vcap import (
    BaseBackend,
    DETECTION_NODE_TYPE,
    DetectionNode,
    OPTION_TYPE,
    BaseStreamState
)


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
        backend_class: Type[BaseBackend],
        args,
        kwargs):
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
    except BaseException as e:
        outgoing.put(e)
        return
    else:
        outgoing.put(None)

    pool = multiprocessing.pool.ThreadPool(processes=num_workers)

    def handle_request(request: RpcRequest):
        try:
            fn = getattr(backend, request.function)  # noqa: F821
            result = fn(*request.args, **request.kwargs)

            if request.function == "process_frame":
                # Special case for 'process_frame': We return the input nodes
                # to the function back to the parent process, in case the
                # capsule modified something in-place.
                result = (request.kwargs["detection_node"], result)
        except BaseException as e:
            result = None
            exception = e
        else:
            exception = None

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
            future = self._futures.pop(response.request_id)
            if response.exception:
                future.set_exception(response.exception)
            else:
                future.set_result(response.result)

    def _rpc_call(self, function: str, *args, **kwargs) -> Any:
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
        self._futures[request.request_id] = future
        self._outgoing.put(request)
        return future.result()

    def process_frame(self,
                      frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) \
            -> DETECTION_NODE_TYPE:
        """Sends a frame to be processed by a worker process.
        The detection_node input is carefully kept track of, to see if the
        worker process made any modifications to it.
        """
        # Assign individual detection nodes with a _object_id in case they
        # are modified in the other process, so that they can then be updated
        input_nodes_by_id: Dict[int, DetectionNode] = {}
        for node in self._flatten_node(detection_node):
            node._object_id = id(node)
            input_nodes_by_id[node._object_id] = node

        # Run the process_frame method
        # in_nodes is the nodes that were fed as inputs
        # out_nodes is the nodes the were output by the process_frame method
        in_nodes, out_nodes = self._rpc_call(
            "process_frame",
            frame=frame,
            detection_node=detection_node,
            options=options,
            state=state)

        # Create a flat list of nodes that came from the worker process
        in_nodes_list: List[DetectionNode] = self._flatten_node(in_nodes)
        out_nodes_list: List[DetectionNode] = self._flatten_node(out_nodes)

        # Update any nodes from this process with new information from
        # the worker process.
        return_nodes: List[DetectionNode] = []
        for detection_node in in_nodes_list + out_nodes_list:
            if (not hasattr(detection_node, "_object_id")
                    or detection_node._object_id not in input_nodes_by_id):
                # If this is a new detection node, not one that already existed
                # then there is nothing to 'update'
                if detection_node in out_nodes_list:
                    return_nodes.append(detection_node)
                continue

            input_node = input_nodes_by_id[detection_node._object_id]
            input_node.attributes.update(detection_node.attributes)
            input_node.extra_data.update(detection_node.extra_data)
            if input_node.track_id is None:
                input_node.track_id = detection_node.track_id
            if input_node.encoding is None:
                input_node.encoding = detection_node.encoding

            if detection_node in out_nodes_list:
                # Since the process_frame output one of the nodes that was
                # originally an input, we return the input, not the copy
                return_nodes.append(input_node)

        # Now we try to match the DETECTION_NODE_TYPE from the original
        # process_frame method, in order to be as correct as possible
        if out_nodes is None:
            return None
        elif isinstance(out_nodes, DetectionNode):
            return return_nodes[0]
        else:
            return return_nodes

    @staticmethod
    def _flatten_node(node: DETECTION_NODE_TYPE) -> List[DetectionNode]:
        if node is None:
            return []
        elif isinstance(node, DetectionNode):
            return [node]
        else:
            return list(node)

    def distances(self, *args, **kwargs) -> np.ndarray:
        return self._rpc_call("distances", *args, **kwargs)

    def batch_predict(self, *args, **kwargs) -> NoReturn:
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

        self._incoming.close()
        self._outgoing.close()

        self._incoming.join_thread()
        self._outgoing.join_thread()
