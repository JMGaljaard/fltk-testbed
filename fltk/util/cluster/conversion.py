from pathlib import Path
from typing import Union

from pint import UnitRegistry


class Convert:
    """
    Conversion class, wrapper around pint UnitRegistry. Assumes that the active path is set to the project root.
    Otherwise, provide a custom path to the conversion file when called from a different directory.
    """

    CONVERSION_PATH = Path('configs/quantities/kubernetes.conf')

    def __init__(self, path: Path = None):
        if path:
            self.__Registry = UnitRegistry(filename=str(path)) # pylint: disable=invalid-name
        else:
            self.__Registry = UnitRegistry(filename=str(self.CONVERSION_PATH)) # pylint: disable=invalid-name

    def __call__(self, value: Union[str, int]) -> int:
        """
        Function to convert str representation of a CPU/memory quantity into an integer representation. For conversion
        metrics see `<project_root>/configs/quantities/kubernetes.conf`
        @param value: String representation of CPU/memory to be converted to quantity.
        @type value: str
        @return: Integer representation of CPU/memory quantity that was provided by the caller.
        @rtype: int
        """
        return self.__Registry.Quantity(value)
