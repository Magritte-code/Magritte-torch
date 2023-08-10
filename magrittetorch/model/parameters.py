from typing import Optional, Generic, TypeVar

T = TypeVar('T')
class Parameter(Generic[T]):
    """Parameter class

    Args:
        Generic (_type_): Type of the value of the parameter
    """
    def __init__(self, name : str, value : Optional[T] = None) -> None:
        self.name : str = name
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
    npoints = Parameter[int]("npoints")
    nfreqs = Parameter[int]("nfreqs")
    nrays = Parameter[int]("nrays")
    name = Parameter[str]("name")
    nboundary = Parameter[int]("nboundary")
    #TODO: complete list of parameters

    



    #TODO: first create IO class to handle IO ops
    def read(self, file_name : str) -> None:
        pass
    def write(self, file_name : str) -> None:
        pass
