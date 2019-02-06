"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
"""

import os
import sys
import argparse

from .core.frame import TensorFrame


def main():
    """Test placeholder to run module as a program."""

    progname = 'dummy'

    parser = argparse.ArgumentParser(
        prog=progname,
        description='dummy'
    )

    parser.add_argument('--file', action='store', required=False, help='Input file')

    args = parser.parse_args(sys.argv[1:])

    if args.file != None:
        if not os.path.isfile(args.file):
            sys.exit(progname + ': cant read file \"' + args.file + '\".')


        tensorframe = TensorFrame(memory='128MB')
        pabatch = tensorframe.read_csv(args.file).to_arrow()

        print(pabatch.columns)
        print(pabatch.schema)

        column_sums = tensorframe.sum()
        print('column_sums:' + str(column_sums))
    return 0

if __name__ == '__main__':
    sys.exit(main())
