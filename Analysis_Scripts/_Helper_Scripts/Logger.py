########################################################################################################################
# Apache License 2.0
########################################################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Nina de Lacy

########################################################################################################################
# Import packages
########################################################################################################################
import atexit
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union

########################################################################################################################
# Custom PRINT level — sits between INFO (20) and WARNING (30)
########################################################################################################################
_PRINT_LEVEL: int = 25
logging.addLevelName(_PRINT_LEVEL, 'PRINT')

########################################################################################################################
# Logger class
########################################################################################################################


class Logger:
    """
    logger.print(...)       Timestamped output at PRINT level — mirrors regular print()
    logger.debug(...)       Detailed diagnostic info
    logger.info(...)        Confirmation that things are working as expected
    logger.warning(...)     Unexpected situation, but execution continues
    logger.error(...)       A serious problem occurred
    logger.critical(...)    Program may not be able to continue
    logger.exception(...)   Like error(), but also captures the current traceback
    logger.set_level(...)   Adjust the minimum level shown, e.g. logging.WARNING silences debug/info
    """

    def __init__(self, filename: Optional[Union[str, int]] = 1):
        self.start_time = datetime.now()

        # Resolve filename
        if filename is None:
            log_file = None                                                     # console only, no .log file
        elif filename == 1:
            log_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]      # auto-name from script
        elif isinstance(filename, str):
            log_file = os.path.splitext(filename)[0]                           # use provided name
        else:
            raise TypeError(f"filename must be None, 1, or a string, got {type(filename).__name__!r}")

        # Set up the underlying logger
        self._logger = logging.getLogger(log_file or os.path.splitext(os.path.basename(sys.argv[0]))[0])
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        if log_file is not None:
            file_handler = logging.FileHandler(f"{log_file}.log")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # Opening banner
        self._logger.log(_PRINT_LEVEL, "=" * 100)
        self.print(f"Logging of {os.path.basename(sys.argv[0])}")
        self.print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.print("Logging starts.")
        self._logger.log(_PRINT_LEVEL, "=" * 100)
        atexit.register(self._on_exit)

    # Public API
    def print(self, *args, sep: str = " ") -> None:
        self._logger.log(_PRINT_LEVEL, sep.join(str(a) for a in args))

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)

    def set_level(self, level: int) -> None:
        self._logger.setLevel(level)

    def _on_exit(self) -> None:
        end_time = datetime.now()
        self._logger.log(_PRINT_LEVEL, "=" * 100)
        self.print("Logging ends.")
        self.print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.print(f"Elapsed time: {end_time - self.start_time}")
        self._logger.log(_PRINT_LEVEL, "=" * 100)

########################################################################################################################
# Public function
########################################################################################################################


def create_log(filename: Optional[Union[str, int]] = 1) -> Logger:
    """
    :param filename: None for console only (no .log file);
                     1 to auto-name from the script name (default);
                     a string to use as the .log filename.
    :return: A Logger instance.
    """
    return Logger(filename)

########################################################################################################################
