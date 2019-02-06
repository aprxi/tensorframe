'''
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).

Helper functions
'''
import re

def parse_bytestr(size, minimum=2**10, maximum=2**40):
    '''
    Parse a string of bytes/units (e.g. 8KB, 32MB, 2GB) to a number of bytes
    '''
    def _verify_size(bytesize):
        if bytesize <= (minimum):
            #minimal size 1KB
            return minimum
        if bytesize > (maximum):
            #maximum size 1TB
            raise ValueError('To many bytes:' + str(bytesize) + ',max: ' + str(maximum))
        return bytesize

    if isinstance(size, int):
        return _verify_size(size)

    if isinstance(size, float):
        return _verify_size(int(size))

    if isinstance(size, str):
        available_units = {
            "K": minimum,
            "M": 2**20,
            "G": 2**30,
            "T": maximum,
        }

        size_parsed = list(filter(None, re.split(r'(\d+)', size)))

        if not size_parsed or size_parsed[0].isdigit() is False:
            raise ValueError('str contains invalid data: ' + size)

        if 1 <= size_parsed.__len__() <= 2:
            mult = int(size_parsed[0])

            if size_parsed.__len__() == 1:
                #No unit given, assume bytes and check if its within boundary
                return _verify_size(mult)

            if size_parsed.__len__() == 2:
                unit = size_parsed[1]

                if unit[0] not in available_units:
                    raise ValueError('Size not recognised: ' + size)

                if len(unit) == 1 or (len(unit) == 2 and unit[-1] == 'B'):
                    pass
                else:
                    raise ValueError('Size not recognised: ' + size)

                return _verify_size(mult * available_units[unit[0]])

        raise Exception('invalid path')

    raise ValueError('size type not recognised: ' + str(size))
