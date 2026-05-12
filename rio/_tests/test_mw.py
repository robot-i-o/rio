# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
import random
import unittest
from enum import Enum, auto

import numpy as np
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ClientFactory, ServerFactory, ServerManager
from rio_hw.node import Node


class RequestType(Enum):
    FORWARD = auto()


class PayloadTest(Node):
    """
    Simple node that listens for incoming data and forwards it with timestamps
    """

    __api__ = [
        "read_data",
        "get_all_data",
        "send_data",
    ]
    __pub__ = True
    __req__ = True

    def __init__(
        self,
        dtype=np.float32,
        *,
        freq: int = 500,
        max_buffer_size: int = 10,
        max_queue_size: int = 10,
        payload_size: int = 1024,
        **kwargs,
    ):
        self.dtype = dtype
        self.payload_size = payload_size
        logger.info(f"ForwardingNode initialized with payload size: {self.payload_size} bytes")
        super().__init__(freq=freq, max_buffer_size=max_buffer_size, max_queue_size=max_queue_size, **kwargs)

    def __post_init__(self):
        self.example_request = {
            "type": next(iter(RequestType)).value,
            "data": np.zeros(self.payload_size, dtype=self.dtype),
            "timestamp": time.now(),
            "forward_timestamp": time.now(),
        }
        self.example_data = {
            "payload": np.zeros(self.payload_size, dtype=self.dtype),
            "request_timestamp": time.now(),
            "forward_timestamp": time.now(),
        }
        self.worker = None
        self.run = self.pubreq
        super().__post_init__()

    def pubreq(self):
        try:
            rate = time.Rate(self.freq)
            self.req_ready_event.set()
            # Clear ring buffer before starting
            # self.ring_buffer.clear()

            while not self.exit_event.is_set():
                # Process requests
                try:
                    req = self.request_queue.get()
                    if req and req["type"] == RequestType.FORWARD.value:
                        # Forward data
                        data = {
                            "payload": req["data"],
                            "request_timestamp": req.get("timestamp"),
                            "forward_timestamp": time.now(),
                        }
                        logger.debug("ForwardingNode forwarding data")
                        self.ring_buffer.put(data)
                    else:
                        pass
                except queue.Empty:
                    pass

                rate.precise_sleep()
        except KeyboardInterrupt:
            pass
        finally:
            pass

    def read_data(self, k=None, out=None):
        """Get current or last k states"""
        if hasattr(self.ring_buffer, "count") and self.ring_buffer.count == 0:
            return None
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_data(self):
        """Get all stored states"""
        return self.ring_buffer.get_all()

    def send_data(self, data: bytes, timestamp=None):
        """
        Forward data through the node

        Args:
            data: Payload to forward
            timestamp: Original timestamp (auto-generated if None)
        """
        req = {
            "type": RequestType.FORWARD.value,
            "data": data,
            "timestamp": timestamp if timestamp is not None else time.now(),
        }
        self.request_queue.put(req)


def payload_server_factory(middleware, **kwargs):
    return ServerFactory(middleware, PayloadTest, **kwargs)


def payload_client_factory(middleware, **kwargs):
    return ClientFactory(middleware, PayloadTest, **kwargs)


def create_server_fn(middleware, addr, verbose, timeout, freq):
    return lambda: payload_server_factory(middleware, addr=addr, verbose=verbose, timeout=timeout, freq=freq)


def create_client_fn(middleware, addr, verbose, timeout, freq):
    return lambda: payload_client_factory(middleware, addr=addr, verbose=verbose, timeout=timeout, freq=freq)


class TestMiddleware(unittest.TestCase):
    def setUp(self):
        self.middlewares = ["Zenoh", "Shm", "Thread", "Portal", "ZeroRpc"]
        self.freq = 50
        self.seed = 42
        self._host = "127.0.0.1"
        self._port = 7447
        self.addr = f"{self._host}:{self._port}"
        self.verbose = False
        self.timeout = 1.0

        self.mp_method = "fork"
        mp.set_start_method(self.mp_method, force=True)
        random.seed(self.seed)
        self.ran_gen = np.random.Generator(np.random.PCG64(seed=self.seed))

    def _test_simple_msg(self, middleware):
        """Send simple messages through the middleware"""
        server_fn = create_server_fn(middleware, self.addr, self.verbose, self.timeout, self.freq)
        client_fn = create_client_fn(middleware, self.addr, self.verbose, self.timeout, self.freq)

        with ServerManager(middleware, [server_fn]):
            with client_fn() as payload:
                # Test random float
                mock_value = self.ran_gen.integers(0, 256, size=1024, dtype=np.uint8)
                payload.send_data(mock_value)
                wait = True
                while wait:
                    data = payload.read_data()
                    if data:
                        wait = False
                        break
                retrieved = data["payload"]
                np.testing.assert_array_equal(retrieved, mock_value)

    def test_suite(self):
        for middleware in self.middlewares:
            print(f"Testing middleware: {middleware}")
            with self.subTest(middleware=middleware):
                self._test_simple_msg(middleware)


if __name__ == "__main__":
    unittest.main()
