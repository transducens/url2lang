
# Based on https://aclanthology.org/W16-2366.pdf 4.2

import os
import sys
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import url2lang.utils.utils as utils

import sklearn.metrics
from tldextract import extract
import pycountry

# Disable (less verbose) 3rd party logging
logging.getLogger("filelock").setLevel(logging.WARNING)

# Languages from: https://commoncrawl.github.io/cc-crawl-statistics/plots/languages
# Order is relevant (because of greedy policy): same order as found in source
_langs_to_detect_alpha_3 = [
    "eng", "deu", "rus", "fra", "zho", "spa",
    "jpn", "ita", "nld", "pol", "por", "ces",
    "vie", "tur", "ind", "swe", "ara", "fas",
    "kor", "hun", "ell", "ron", "dan", "fin",
    "tha", "slk", "nor", "ukr", "bul", "cat",
    "srp", "hrv", "slv", "lit", "hin", "est",
    "heb", "lat", "ben", "lav", "msa", "bos",
    "sqi", "tam", "glg", "isl", "aze", "kat",
    "mkd", "eus", "hye", "nep", "urd", "mon",
    "mal", "kaz", "mar", "tel", "nno", "bel",
    "uzb", "guj", "kan", "mya", "khm", "cym",
    "epo", "tgl", "sin", "afr", "tat", "swa",
    "gle", "pan", "kur", "kir", "tgk", "mlt",
    "fao", "ori", "lao", "som", "ltz", "oci",
    "amh", "fry", "bak", "pus", "san", "bre",
    "mlg", "hau", "tuk", "war", "asm", "cos",
    "div", "jav", "ceb", "kin", "hat", "zul",
    "gla", "bod", "xho", "yid", "snd", "mri",
    "uig", "roh", "sun", "kal", "yor", "tir",
    "abk", "bih", "haw", "hmn", "ina", "que",
    "grn", "ibo", "nya", "sco", "sna", "sot",
    "smo", "vol", "glv", "orm", "ile", "syr",
    "aar", "dzo", "iku", "kha", "lin", "lug",
    "mfe", "aka", "aym", "bis", "chr", "crs",
    "fij", "ipk", "nso", "run", "sag", "ssw",
    "ton", "tsn", "wol", "zha", "got", "kas",
    "lif", "nau", "sux", "tso", "ven"]
_langs_to_detect = ["unknown"]
_3_letter_to_2_letter = True # Look for 2-letter code in URLs
_3_letter_to_2_letter_force = False # Do not add other thing which is not 2-letter code

def global_preprocessing():
    global _langs_to_detect

    initial_languages_skip = len(_langs_to_detect)

    # Get languages which will be detected in URLs
    for _lang in _langs_to_detect_alpha_3:
        if _3_letter_to_2_letter:
            lang_alpha_2 = pycountry.languages.get(alpha_3=_lang)

            if "alpha_2" in dir(lang_alpha_2) and lang_alpha_2.alpha_2 is not None:
                lang_alpha_2 = lang_alpha_2.alpha_2

                if lang_alpha_2 not in _langs_to_detect:
                    _langs_to_detect.append(lang_alpha_2)
                else:
                    logging.debug("Language %s already loaded: %s", _lang, lang_alpha_2)
            else:
                if _3_letter_to_2_letter_force:
                    if lang_alpha_2 is None or "alpha_2" not in dir(lang_alpha_2) or lang_alpha_2.alpha_2 is None:
                        logging.error("Language %s couldn't be processed", _lang)
                    else:
                        logging.error("Language %s couldn't be processed: %s", lang_alpha_2, _lang)
                else:
                    logging.warning("Language %s: couldn't get 2-letter form: using initial value", _lang)

                    _langs_to_detect.append(_lang)
        else:
            _langs_to_detect.append(_lang)

    logging.debug("%d languages loaded (initial languages: %d): %s",
                  len(_langs_to_detect) - initial_languages_skip, len(_langs_to_detect_alpha_3), ", ".join(_langs_to_detect))

def get_gs(file):
    gs, url2lang, lang2url = set(), {}, {}

    for idx, line in enumerate(file, 1):
        line = line.rstrip("\r\n").split('\t')

        if len(line) != 2:
            logging.warning("GS: unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        url, lang = line

        gs.add('\t'.join(line))

        if url in url2lang:
            logging.warning("GS: duplicated URL in TSV entry #%d: %s", idx, url)

            if url2lang[url] != lang:
                logging.error("GS: duplicated URL: different lang: got %s, but %s had been registred before", lang, url2lang[url])

            continue

        if lang not in lang2url:
            lang2url[lang] = set()

        if url in lang2url[lang]:
            logging.warning("GS: duplicated URL: already registred in lang %s", lang)

            continue

        url2lang[url] = lang
        lang2url[lang].add(url)

    return gs, url2lang, lang2url

def evaluate(urls, gs, gs_url2lang, gs_lang2url, lowercase=False, print_pairs=True,
             print_negative_matches=False, print_score=False):
    y_pred, y_true = [], []
    matches = 0
    gs_provided = False if len(gs) == 0 else True

    for url in urls:
        detected_lang = "unknown"
        _url = url.lower() if lowercase else url
        importance2lang = {1: [], 2: [], 3: [], 4: []} # Hierarchy: resource variables, subdomain, directory, public suffix
        subdomain, domain, public_suffix = extract(_url, include_psl_private_domains=False)

        # Detect lang
        for lang2check in [lang for lang in _langs_to_detect if lang != "unknown"]:
            if f"lang={lang2check}" in _url or f"language={lang2check}" in _url:
                importance2lang[1].append(lang2check)

            if subdomain and subdomain == lang2check:
                importance2lang[2].append(lang2check)

            if f"/{lang2check}/" in _url or _url.endswith(f"/{lang2check}"):
                importance2lang[3].append(lang2check)

            if public_suffix and public_suffix == lang2check:
                importance2lang[4].append(lang2check)

        # Check if any language was detected
        for importance in sorted(importance2lang.keys()):
            langs = importance2lang[importance]

            if len(langs) != 0 and detected_lang == "unknown":
                detected_lang = langs[0] # Greedy policy: get first detected language with highest importance

            logging.debug("Detected languages with importance %d: %s", importance, langs)

        if detected_lang != "unknown":
            matches += 1

        # Eval
        if gs_provided:
            if url in gs_url2lang:
                # We only evaluate URLs which are present in the GS
                y_true.append(gs_url2lang[url])
                y_pred.append(detected_lang)
            else:
                logging.error("Evaluated URL not present in the GS: this will falsify the evaluation: %s", url)

        # Print results?
        if print_pairs and (print_negative_matches or detected_lang != "unknown"):
            if print_score:
                print(f"{url}\t{detected_lang}")
            else:
                print(url)

    return y_pred, y_true, matches

def main(args):
    input_file = args.input
    gs_file = args.gold_standard
    lowercase = args.lowercase
    print_negative_matches = args.print_negative_matches
    print_score = args.print_score

    gs, gs_url2lang, gs_lang2url = get_gs(gs_file) if gs_file else (set(), {}, {})
    urls = []

    # Read URLs
    for idx, line in enumerate(input_file, 1):
        line = line.rstrip("\r\n").split('\t')

        if len(line) != 1:
            logging.warning("Unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 1, len(line))

            continue

        url = line[0]

        urls.append(url)

    logging.info("Provided URLs: %d", len(urls))

    # Evaluate
    logging.info("Evaluating...")

    y_pred, y_true, matches =\
        evaluate(urls, gs, gs_url2lang, gs_lang2url, lowercase=lowercase,
                 print_negative_matches=print_negative_matches, print_score=print_score)

    # Some statistics
    negative_matches = len(urls) - matches

    logging.info("Positive and negative (i.e. language not detected) matches: %d, %d", matches, negative_matches)

    if gs_file:
        logging.info("Using GS in order to get some evaluation metrics. Langs order: %s", str(_langs_to_detect))

        # Log metrics
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=_langs_to_detect)

        logging.info("GS: confusion matrix: %s", list(confusion_matrix))

        for lang in _langs_to_detect:
            precision = sklearn.metrics.precision_score(y_true, y_pred, labels=_langs_to_detect, pos_label=lang)
            recall = sklearn.metrics.recall_score(y_true, y_pred, labels=_langs_to_detect, pos_label=lang)
            f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=_langs_to_detect, pos_label=lang)

            logging.info("GS: lang %s: precision: %s", lang, precision)
            logging.info("GS: lang %s: recall: %s", lang, recall)
            logging.info("GS: lang %s: F1: %s", lang, f1)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Detect document lang using only the URL")

    parser.add_argument('input', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with URLs")

    parser.add_argument('--gold-standard', type=argparse.FileType('rt', errors="backslashreplace"), help="GS filename with URLs and lang (TSV format)")
    parser.add_argument('--lowercase', action="store_true", help="Lowercase URLs (GS as well if provided). It might increase the evaluation results")
    parser.add_argument('--print-negative-matches', action="store_true",
                        help="Print negative matches (i.e. not only possitive matches)")
    parser.add_argument('--print-score', action="store_true",
                        help="Print 0 or 1 for positive or negative matches, respectively")

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
