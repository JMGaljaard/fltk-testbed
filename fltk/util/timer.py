import time
from contextlib import contextmanager
# from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = time.time()
    elapser = lambda: time.time() - start
    yield lambda: elapser()
    end = time.time()
    elapser = lambda: end-start