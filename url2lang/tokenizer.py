
import os
import sys
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/..")

import url2lang.utils.utils as utils

#from nltk.tokenize import RegexpTokenizer
import nltk.tokenize

_tokenizer_regex = r'[^\W_0-9]+|[^\w\s]+|_+|\s+|[0-9]+' # Similar to wordpunct_tokenize
_tokenize = nltk.tokenize.RegexpTokenizer(_tokenizer_regex).tokenize
_tokenize_gaps = nltk.tokenize.RegexpTokenizer(_tokenizer_regex, gaps=True).tokenize

_tokenizers = {
    "tokenizers": {
        "wordpunct_tokenize_urls": _tokenize,
        "word_tokenize": nltk.tokenize.word_tokenize
    },
    "gaps": {
        "wordpunct_tokenize_urls": _tokenize_gaps
    }
}

def tokenize(s, check_gaps=True, gaps_whitelist=[' '], tokenizer="wordpunct_tokenize_urls"):
    tokenized_str = _tokenizers["tokenizers"][tokenizer](s)

    if check_gaps:
        try:
            gaps_tokenizer = _tokenizers["gaps"][tokenizer]
        except KeyError:
            # Tokenizer without gaps support
            gaps_tokenizer = lambda s: [] # Fake "no gaps"

        tokenized_str_gaps = gaps_tokenizer(s)

        if len(tokenized_str_gaps) != 0:
            for gap in tokenized_str_gaps:
                if gap not in gaps_whitelist:
                    logging.error("Found gaps tokenizing, but the tokenizer should be complete (bug?): %s", s)
                    logging.error("Gaps: %d: %s", len(tokenized_str_gaps), str(tokenized_str_gaps))

                    break

    return tokenized_str

def main(args):
    input_filename = args.input_filename
    check_gaps = not args.disable_check_gaps
    tokenizer = args.tokenizer
    disable_join_tokens = args.disable_join_tokens

    join_function = (lambda s: s) if disable_join_tokens else ' '.join

    for s in input_filename:
        print(join_function(tokenize(s.rstrip('\n'), check_gaps=check_gaps, tokenizer=tokenizer)))

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Tokenizer")

    # We can't use argparse.FileType('rb') due to https://bugs.python.org/issue14156 -> Workaround: use 'errors="backslashreplace"'
    parser.add_argument('--input-filename', type=argparse.FileType('rt', errors="backslashreplace"), default='-', help="Input file with sentences to be tokenized")
    parser.add_argument('--disable-check-gaps', action="store_true", help="Do not check gaps when tokenize provided sentences")
    parser.add_argument('--tokenizer', choices=["wordpunct_tokenize_urls", "word_tokenize"], default="wordpunct_tokenize_urls",
                        help="Different available tokenizers")
    parser.add_argument('--disable-join-tokens', action="store_true", help="Do not join the sentences with ' ' but show the list with the tokens")

    parser.add_argument('-v', '--verbose', action="store_true", help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug("Arguments processed: {}".format(str(args)))

    main(args)
