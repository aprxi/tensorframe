'''
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).

Helper functions
'''
import re

def parse_bytestr(size):
    '''
    Parse a string of bytes/units (e.g. 8KB, 32MB, 2GB) to a number of bytes
    '''

    minimum = 2**10
    maximum = 2**40

    def _verify_size(bytesize):
        if bytesize <= (minimum):
            #minimal size 1KB
            return minimum
        if bytesize > (maximum):
            #maximum size 1TB
            raise ValueError('To many bytes:' + str(bytesize) + ',max: ' + str(maximum))
        return bytesize


    if isinstance(size, str):
        available_units = {
            "K": minimum,
            "M": 2**20,
            "G": 2**30,
            "T": maximum,
        }

        size_parsed = list(filter(None, re.split(r'(\d+)', size)))

        if not size_parsed or size_parsed[0].isdigit() is False:
            raise ValueError('Size not recognised: ' + size)

        mult = int(size_parsed[0])

        if len(size_parsed) == 1:
            #No unit given, assume bytes and check if its within boundary
            return _verify_size(mult)

        if len(size_parsed) == 2:
            unit = size_parsed[1]

            if unit[0] not in available_units:
                raise ValueError('Size not recognised: ' + size)

            if len(unit) == 1 or (len(unit) == 2 and unit[-1] == 'B'):
                pass
            else:
                raise ValueError('Size not recognised: ' + size)

            return _verify_size(mult * available_units[unit[0]])

    if isinstance(size, int):
        return _verify_size(mult)
    if isinstance(size, float):
        return _verify_size(int(mult))

    raise ValueError('Size not recognised: ' + size)


# TODO: datastr parser not yet finished
#def parse_datestr(date):
#    '''
#    Parse datestring using accelerated library
#    '''
#    if not isinstance(date, str):
#        raise TypeError
#
#    #_loadmod_libparser()
#    result = libparser.parseDateStr()
#    return result


#def parse_datestr_classic(date):
#    '''
#    Parse datestring using classic python dateutil library
#    '''
#    if not isinstance(date, str):
#        raise TypeError
#
#    from dateutil.parser import parse
#    #_loadmod_libparser()
#
#    datestamp = parse(date)
#
#    #dateutil.parser.parse('05.01.2015', dayfirst=True)
#    result = datestamp.timestamp()
#    return result

