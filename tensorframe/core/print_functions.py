"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

from sys import stderr


def error(*args, **kwargs):
    """Prints an ERROR statement to stderr, complemented with input."""
    print('ERROR:', file=stderr, end='')
    return print(*args, file=stderr, **kwargs)


def warning(*args, **kwargs):
    """Prints a WARNING statement to stderr, complemented with input."""
    print('WARNING:', file=stderr, end='')
    return print(*args, file=stderr, **kwargs)
