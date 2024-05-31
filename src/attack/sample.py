"""Sample docstring for Generator."""

from typing import Generator


def f() -> Generator[int]:
    """Function that always yields 1.

    Returns:
        something bad (float): something.
    """
    while True:
        yield 1
