import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch


class do_multi_stream_local(threading.local):

    def __init__(self):
        self.do_multi_stream = False


_local = do_multi_stream_local()


def set_do_multi_stream(enable: bool):
    _local.do_multi_stream = enable


def do_multi_stream() -> bool:
    return _local.do_multi_stream


@contextmanager
def with_multi_stream(enable: bool):
    prev_do_multi_stream = _local.do_multi_stream
    set_do_multi_stream(enable)
    try:
        yield
    finally:
        set_do_multi_stream(prev_do_multi_stream)


def maybe_execute_in_parallel(
        fn0: Callable,
        fn1: Callable,
        event0: torch.cuda.Event,
        event1: torch.cuda.Event,
        aux_stream: Optional[torch.cuda.Stream] = None) -> tuple[Any, Any]:
    """Utility function to run two functions in two cuda streams in parallel. Multi-stream is
    only enabled when cuda graph is turned on because switch stream has extra host overhead.

    This design is mainly for low latency use case. It needs to be improved for max throughput
    use case.
    For simplicity, fn0 and fn1 do not support inputs.

    Args:
        fn0 (Callable): callable for the default stream
        fn1 (Callable): callable for the second stream, aux_stream
        event0 (torch.cuda.Event): cuda event for fn0
        event1 (torch.cuda.Event): cuda event for fn1
        aux_stream (Optional[torch.cuda.Stream]): the second cuda stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        tuple[Any, Any]: the return values of fn0() and fn1()
    """

    multi_stream = do_multi_stream() and aux_stream is not None

    # Debug: Print multi-stream decision
    import os
    if os.environ.get('DEBUG_MULTISTREAM') == '1':
        print(f"\n=== maybe_execute_in_parallel ===")
        print(f"do_multi_stream(): {do_multi_stream()}")
        print(f"aux_stream: {aux_stream}")
        if aux_stream:
            print(f"aux_stream.cuda_stream: {aux_stream.cuda_stream}")
        print(f"multi_stream decision: {multi_stream}")
        print(f"Current stream: {torch.cuda.current_stream().cuda_stream}")

    if multi_stream:
        event0.record()
        result0 = fn0()

        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.wait()
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)
