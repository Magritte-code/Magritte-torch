from enum import Enum

# The dev might want to have some additional information printed to the console, which can be irrelevant for the user.
# This class helps filtering out the relevant info.

class Level(Enum):
    """Enum for the importance level of a message
    """
    DEBUG = 0#: Debugging information
    INFO = 1#: Information
    WARNING = 2#: Warning messages
    ERROR = 3#: Error messages

class Logger:
    default_level = 2 #: Can be set by the user/dev to control the verbosity of the logger
    def __init__(self, level = None):
        if level is None:
            self.level = self.default_level
        else:
            self.level = level

    def log(self, message:str , level: Level = Level.INFO):
        """Prints the message to the console if the level of the message is higher than the level of the logger

        Args:
            level (Level, optional): Importance level of the message. Defaults to Level.INFO.
            message (str): message to be printed
        """
        if level.value >= self.level:
            print(f"{level.name}: {message}")