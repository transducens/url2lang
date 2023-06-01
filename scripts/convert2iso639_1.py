
import os
import sys
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import url2lang.utils.utils as utils

import pycountry

# Disable (less verbose) 3rd party logging
logging.getLogger("filelock").setLevel(logging.WARNING)

def main(args):
    input_file = args.input
    column = args.column

    for idx, line in enumerate(input_file, 1):
        line = line.rstrip("\r\n").split('\t')
        lang = line[column]

        try:
            _lang = pycountry.languages.get(alpha_3=lang).alpha_2

            if _lang:
                lang = _lang
        except Exception as e:
            loggin.warning("Couldn't process lang %s: line #%d", lang, idx)

        line[column] = lang

        print('\t'.join(line))

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Convert lang code to ISO 639-1")

    parser.add_argument('input', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with data")

    parser.add_argument('--column', type=int, default=0, help="Column of the data which will be converted")
    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    global_preprocessing()

    main(args)
