from time import perf_counter
from datetime import datetime

class Timer ():
    """
    A simple timer class.
    """
    def __init__ (self, name: str):
        """
        Set a name for the times.

        Parameters
        ---
        name : str
            Name of the timer.
        """
        self.name      = name
        self.starts: list[float]    = []
        self.stops: list[float]     = []
        self.intervals: list[float] = []
        self.total     = 0.0
    def start (self) -> None:
        """
        Start the timer.
        """
        self.starts.append(perf_counter())
    def stop (self) -> None:
        """
        Stop the timer.
        """
        self.stops.append(perf_counter())
        self.intervals.append(self.stops[-1]-self.starts[-1])
        self.total += self.intervals[-1]
    def print (self) -> str:
        """
        Print the elapsed time.
        """
        return f'timer: {self.name} = {self.total}'

def timestamp() -> str:
    """
    Returns a time stamp for the current date and time.

    Returns
    -------
    out : str
        A string containing the current date and time.
    """
    return datetime.now().strftime('%y%m%d-%H%M%S')