"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

import json


def serialize_as_string(value):
    """Convert input value to string using __str__() method of itself."""
    return value.__str__()


def write_json_serialized(filename, dictionary):
    """Write a dictionary to file in json-format, with its values serialized to a string."""
    assert isinstance(filename, str)
    assert isinstance(dictionary, dict)

    with open(filename, 'w') as outfile:
        json.dump(dictionary, outfile, default=serialize_as_string)
    return True
