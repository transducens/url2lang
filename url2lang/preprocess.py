
import re
import logging
import urllib.parse

import url2lang.utils.utils as utils
from url2lang.tokenizer import tokenize

logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("url2lang")

def remove_blanks(url):
    url = re.sub(r'\s+', ' ', url)
    url = re.sub(r'^\s+|\s+$', '', url)

    return url

_stringify_url_replace_chars = ['.', '-', '_', '=', '?', '\n', '\r', '\t']
def stringify_url(url, separator=' '):
    url = url.split('/')
    url = list(map(lambda u: utils.replace_multiple(u, _stringify_url_replace_chars).strip(), url))
    #url = [' '.join([s for s in u.split(' ') if s != '']) for u in url] # Remove multiple ' '
    url = separator.join(url)
    # Remove blanks
    url = remove_blanks(url)

    return url

def preprocess_url(url, remove_protocol_and_authority=False, remove_positional_data=False, separator=' ',
                   stringify_instead_of_tokenization=False, remove_protocol=True, lower=False,
                   tokenization=True, start_urls_idx=None, end_urls_idx=None, erase_blanks=True):
    if isinstance(url, str):
        url = [url]

        if start_urls_idx is not None or end_urls_idx is not None:
            logger.warning("Provided URL is not a list, but a string, and start:end idxs were provided: "
                           "URL will be split and a substring will be the result instead of a set of URLs")

    start_urls_idx = 0 if start_urls_idx is None else start_urls_idx
    end_urls_idx = len(url) if end_urls_idx is None else end_urls_idx

    if remove_protocol_and_authority:
        if not remove_protocol:
            logging.warning("'remove_protocol' is not True, but since 'remove_protocol_and_authority' is True, it will enabled")

        remove_protocol = True # Just for logic, but it will have no effect

    urls = [u for u in url[0:start_urls_idx]] # Append all initial elements from the provided data

    for u in url[start_urls_idx:end_urls_idx]:
        u = u.rstrip('/')

        if remove_protocol_and_authority:
            u = u[utils.get_idx_resource(u):]
        elif remove_protocol:
            u = u[utils.get_idx_after_protocol(u):]

        if remove_positional_data:
            # e.g. https://www.example.com/resource#position -> https://www.example.com/resource

            ur = u.split('/')
            h = ur[-1].find('#')

            if h != -1:
                ur[-1] = ur[-1][:h]

            u = '/'.join(ur)

        u = urllib.parse.unquote(u, errors="backslashreplace") # WARNING! It is necessary to replace, at least, \t

        if lower:
            u = u.lower()

        # TODO TBD stringify instead of tokenize or stringify after tokenization
        if stringify_instead_of_tokenization:
            u = stringify_url(u, separator=separator)
        elif tokenization:
            u = u.replace('/', separator)
            # Remove blanks
            u = remove_blanks(u)
            # Tokenize
            u = ' '.join(tokenize(u))
        elif erase_blanks:
            u = remove_blanks(u)

        urls.append(u)

    # Append all initial elements from the provided data
    for u in url[end_urls_idx:]:
        urls.append(u)

    return urls
