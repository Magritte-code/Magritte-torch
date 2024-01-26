from typing import Optional, Generic, TypeVar, Iterator, Tuple, Callable, Any
from enum import Enum

T = TypeVar('T')
# TODO: maybe change compile time of type checking to runtime checking; hmm, think about performance implications
#TODO: write docs
class Parameter(Generic[T]):
    """Parameter class

    Args:
        Generic (_type_): Type of the value of the parameter
    """
    def __init__(self, name: str, legacy_converter: Optional[Tuple[str, Optional[Callable[[Any], T]]]] = None) -> None:
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
        # print("setting param", str(self), "to value", value)
        if self.value == None:
            self.value = value
            return
        if self.value != value:
            raise ValueError("The value for parameter " + self.name + " has already been set at: " + str(self.value) + ". Attempted to set at new value: " + str(value))
    
    def get(self) -> T:
        """Returns the value of the parameter

        Raises:
            ValueError: If the parameter has not yet been set

        Returns:
            T: The value of the parameter
        """
        if self.value is None:
            raise ValueError("The value for parameter " + self.name + " has not yet been set.")
        else:
            return self.value

TYPE = TypeVar('TYPE', bound = type[Enum])
class EnumParameter(Parameter[T], Generic[T, TYPE]):
    def __init__(self, enum_type: TYPE, name: str, legacy_converter: Optional[Tuple[str, Optional[Callable[[Any], T]]]] = None) -> None:
        """EnumParameter class
        Due to h5py not natively storing Enums, we need to additionally store the type of Enum for converting

        Args:
            enum_type (TYPE): Type of the Enum
            name (str): Name of the parameter
            legacy_converter (Optional[Tuple[str, Optional[Callable[[Any], T]]]], optional): TODO: also write description for Parameter. Defaults to None.
        """
        Parameter.__init__(self, name, legacy_converter)
        self.enum_type: TYPE = enum_type
    
    def get_enum_type(self) -> TYPE:
        return self.enum_type


class Parameters:
    """List of Parameter's
        pass

    Yields:
        Parameter: parameters for a model
    """

    def __init__(self) -> None:
        self.npoints = Parameter[int]("npoints")
        self.nrays = Parameter[int]("nrays")
        self.nboundary = Parameter[int]("nboundary")
        self.nspecs = Parameter[int]("nspecs")
        self.nlspecs = Parameter[int]("nlspecs")

    def __iter__(self) -> Iterator[Parameter[T]]:
        """Iterator for the parameters of a model

        Yields:
            Iterator[Parameter[T]]: parameters for a model
        """
        for attr in vars(self):
            if not attr.startswith("__"):#no builtin attributes, nor __iter__ itself
                yield self.__getattribute__(attr)
