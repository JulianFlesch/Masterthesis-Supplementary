import cProfile

from contextlib import contextmanager
from timeit import default_timer, repeat


def measure_runtime(callback=lambda: (), repeats=5, number=1):
    """
    Wrapper for `timeit.repeat` function, that measures the walltime for
    executing the callback.
    
    Excerpt from [timeit docu](https://docs.python.org/3/library/timeit.html):
        'Note It’s tempting to calculate mean and standard deviation from 
        the result vector and report these. However, this is not very useful.
        In a typical case, the lowest value gives a lower bound for how fast
        your machine can run the given code snippet; higher values in the result
        vector are typically not caused by variability in Python’s speed, but
        by other processes interfering with your timing accuracy. 
        So the min() of the result is probably the only number you should be
        interested in. After that, you should look at the entire vector and
        apply common sense rather than statistics.'

    
    @return tuple of (min timing, avg timing)
    """

    timings = repeat(callback, repeats=repeats, number=1)

    return (min(timings), sum(timings)/len(timings))


def run_profile(callback_string: str=lambda: ()):
    """
    Wrapper for CProfile function
    """
    assert(type(callback_string) == str)
    return cProfile.run(callback_string)


@contextmanager
def elapsed_timer():
    """
    Creates a with-statement context that notes the starttime and yields
    the elapsed time whenever called

    use: 
        ```
        with elapsed_timer() as t:
            # do something
            print(t())

            # do something
            print(t())
            ...

        ```
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
