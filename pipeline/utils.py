import time
from typing import Any

nesting_level = 0
is_start = None


def timeit(method):  # decorator for timing functions
    def timed(*args, **kw):
        global is_start
        global nesting_level

        is_start = True
        log("Start {}.".format(method.__name__))
        nesting_level += 1

        start_time = time.time()

        # try:
        result = method(*args, **kw)
        nesting_level -= 1
        # except Exception:
        #     result = None
        #     nesting_level = 0

        end_time = time.time()

        log("End {}. Time: {:0.2f} sec.".format(method.__name__, end_time - start_time))
        
        if is_start:
            print()
            
        is_start = False

        return result

    return timed


def log(entry: Any): 
    global nesting_level
    space = "." * (4 * nesting_level)
    print("{}{}".format(space, entry))
