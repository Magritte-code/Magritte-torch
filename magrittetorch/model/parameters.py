from typing import Optional, Generic, TypeVar, Iterator, Tuple, Callable, Any

T = TypeVar('T')
# TODO: maybe change compile time of type checking to runtime checking; hmm, think about performance implications
#TODO: write docs
class Parameter(Generic[T]):
    """Parameter class

    Args:
        Generic (_type_): Type of the value of the parameter
    """
    def __init__(self, name : str, legacy_converter : Optional[Tuple[str, Optional[Callable[[Any], T]]]] = None) -> None:
        self.name: str = name
        self.legacy_name: str = self.name
        self.legacy_conversion_function: Optional[Callable[[Any], T]] = None
        if legacy_converter is not None:
            self.legacy_name = legacy_converter[0]
            self.legacy_conversion_function = legacy_converter[1]
        self.value : Optional[T] = None
    
    def __str__(self) -> str:
        return "Parameter " + self.name + " with current value: " + str(self.value)
    
    def __repr__(self) -> str:
        return self.name + ": " + str(self.value)

    
    def set_force_overwrite(self, value : T) -> None:
        """Force overwrite the value of the parameter. This can break stuff.

        Args:
            value (T): New value of the parameter
        """
        self.value = value

    def set(self, value : T) -> None:
        """Setter for the value. Returns a ValueError if an old value has already been set and the new value is not equal to the old value.

        Args:
            value (T): New value for parameter
        """
        if self.value == None:
            self.value = value
            return
        if self.value != value:
            raise ValueError("The value for parameter " + self.name + " has already been set at: " + str(self.value) + ". Attempted to set at new value: " + str(value))
    
    def get(self) -> T:
        if self.value is None:
            raise ValueError("The value for parameter " + self.name + " has not yet been set.")
        else:
            return self.value

class Parameters:
    """List of Parameter's

    Yields:
        Parameter: parameters for a model
    """
    npoints = Parameter[int]("npoints")
    # nfreqs = Parameter[int]("nfreqs")
    nrays = Parameter[int]("nrays")
    # name = Parameter[str]("name")
    nboundary = Parameter[int]("nboundary")
    nspecs = Parameter[int]("nspecs")
    nlspecs = Parameter[int]("nlspecs")
    #TODO: complete list of parameters

    def __iter__(self) -> Iterator[Parameter[T]]:
        """Iterator for the parameters of a model

        Yields:
            Iterator[Parameter[T]]: parameters for a model
        """
        for attr in dir(self):
            if not attr.startswith("__"):#no builtin attributes, nor __iter__ itself
                yield self.__getattribute__(attr)

    



    #TODO: first create IO class to handle IO ops
    #TODO: do we actually need to read and write these; only for compatibility reasons with C++ magritte
    # def read(self, file_name : str) -> None:
    #     pass
    # def write(self, file_name : str) -> None:
    #     pass
