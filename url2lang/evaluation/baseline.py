
import os
import sys
import logging
import argparse
import urllib.parse as urlparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import url2lang.url2lang as url2lang_imp
import url2lang.utils.utils as utils

import sklearn.metrics
from tldextract import extract
import pycountry

# Disable (less verbose) 3rd party logging
logging.getLogger("filelock").setLevel(logging.WARNING)

# Order is relevant (because of greedy policy): same order as found in source
_langs_to_detect_alpha_3 = url2lang_imp._langs_to_detect_alpha_3
_unknown_lang_label = url2lang_imp._unknown_lang_label
_langs_to_detect = [_unknown_lang_label] + _langs_to_detect_alpha_3

_check_2_letter_too = True
_check_lang_name_too = False

logger = logging.getLogger("url2lang.baseline")

_warning_once_done = False
def get_gs(file):
    def unk_lang(lang):
        global _warning_once_done

        if lang == _unknown_lang_label:
            # Skip URLs whose lang is unknown in the GS

            if not _warning_once_done:
                _warning_once_done = True

                logger.warning("GS: URLs whose lang is %s are going to be ignored", lang)

            return True

        return False

    gs, url2lang, lang2url = set(), {}, {}

    for idx, line in enumerate(file, 1):
        line = line.rstrip("\r\n").split('\t')

        if len(line) != 2:
            logger.warning("GS: unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        url, lang = line

        if unk_lang(lang):
            continue

        if len(lang) != 3:
            logger.warning("GS: URL lang (lang: %s) was expected to be ISO 639-2: entry #%d", lang, idx)

            continue

        if unk_lang(lang):
            continue

        if lang not in _langs_to_detect:
            logger.warning("GS: URL language (lang: %s) not in the list of languages to be detected: entry #%d", lang, idx)

            continue

        gs.add('\t'.join(line))

        if url in url2lang:
            logger.warning("GS: duplicated URL in TSV entry #%d: %s", idx, url)

            if url2lang[url] != lang:
                logger.error("GS: duplicated URL: different lang: got %s, but %s had been registred before", lang, url2lang[url])

            continue

        if lang not in lang2url:
            lang2url[lang] = set()

        if url in lang2url[lang]:
            logger.warning("GS: duplicated URL: already registred in lang %s", lang)

            continue

        url2lang[url] = lang
        lang2url[lang].add(url)

    return gs, url2lang, lang2url

_importance_hierarchy = [1, 4, 2, 3]    # variables, subdomain, directory and public suffix
                                        # Specific order: variables, directory, public suffix and subdomain
                                        # Research purposes: which order is the best? Set using U2L_IMPORTANCE_HIERARCHY envvar
def evaluate(urls, gs, gs_url2lang, gs_lang2url, lowercase=False, print_pairs=True,
             print_negative_matches=False, print_score=False):
    global _importance_hierarchy

    if "U2L_IMPORTANCE_HIERARCHY" in os.environ:
        _importance_hierarchy_tmp = os.environ["U2L_IMPORTANCE_HIERARCHY"].rstrip(" \r\n").split(',')

        if len(_importance_hierarchy_tmp) != len(_importance_hierarchy):
            logger.warning("Unexpected envvar format: %d fields split by comma were expected: U2L_IMPORTANCE_HIERARCHY", len(_importance_hierarchy))
        else:
            try:
                _importance_hierarchy_tmp = list(map(int, _importance_hierarchy_tmp))
                _importance_hierarchy = _importance_hierarchy_tmp

                logger.debug("_importance_hierarchy has been modified")
            except:
                logger.warning("Unexpected envvar format: int values were expected: U2L_IMPORTANCE_HIERARCHY")

    logger.debug("_importance_hierarchy: %s", ", ".join(map(str, _importance_hierarchy)))

    y_pred, y_true = [], []
    matches = 0
    gs_provided = False if len(gs) == 0 else True

    for url in urls:
        detected_lang = _unknown_lang_label
        _url = url.lower() if lowercase else url
        importance2lang = {1: [], 2: [], 3: [], 4: []} # Hierarchy: resource variables, subdomain, directory, public suffix
        subdomain, domain, public_suffix = extract(_url, include_psl_private_domains=False)

        if len(importance2lang.keys()) != len(_importance_hierarchy):
            raise Exception("Bug: importance2lang length != _importance_hierarchy length")
        if len(set.intersection(set(importance2lang.keys()), set(_importance_hierarchy))) != len(importance2lang.keys()):
            raise Exception("Bug: the defined 'importance' levels are not the same")

        # Detect lang
        for lang2check in [lang for lang in _langs_to_detect if lang != _unknown_lang_label]:
            all_langs2check = [lang2check] # All langs "related" to lang2check (e.g. 2-letter code, full language name)
            lang_pyc = pycountry.languages.get(alpha_3=lang2check)

            if lang_pyc is not None:
                if _check_2_letter_too:
                    if "alpha_2" in dir(lang_pyc) and lang_pyc.alpha_2 is not None:
                        _tmp_lang = lang_pyc.alpha_2

                        if _tmp_lang and len(_tmp_lang) == 2:
                            all_langs2check.append(_tmp_lang)
                    # else: we don't need a warning: best effort approach
                if _check_lang_name_too:
                    if "name" in dir(lang_pyc) and lang_pyc.name is not None:
                        _tmp_lang = str.lower(lang_pyc.name)

                        if _tmp_lang:
                            all_langs2check.append(_tmp_lang)
                    # else: we don't need a warning: best effort approach

            _url_parsed = urlparse.urlparse(_url, allow_fragments=False)
            _url_variables = urlparse.parse_qs(_url_parsed.query)

            for _check in all_langs2check:
                for params in _url_variables.values():
                    for p in params:
                        if p == _check:
                            importance2lang[_importance_hierarchy[0]].append(lang2check)

                if subdomain and subdomain == _check:
                    importance2lang[_importance_hierarchy[1]].append(lang2check)

                if f"/{_check}/" in _url or _url.endswith(f"/{_check}"):
                    importance2lang[_importance_hierarchy[2]].append(lang2check)

                if public_suffix and public_suffix == _check:
                    importance2lang[_importance_hierarchy[3]].append(lang2check)

        # Check if any language was detected
        for importance in sorted(importance2lang.keys()):
            langs = importance2lang[importance]

            if len(langs) != 0 and detected_lang == _unknown_lang_label:
                detected_lang = langs[0] # Greedy policy: get first detected language with highest importance

            if len(langs) != 0:
                logger.debug("Detected languages with importance %d: %s (url: %s)", importance, langs, url)

        if detected_lang != _unknown_lang_label:
            matches += 1

        # Eval
        if gs_provided:
            if url in gs_url2lang:
                # We only evaluate URLs which are present in the GS

                if gs_url2lang[url] in _langs_to_detect:
                    y_true.append(gs_url2lang[url])
                    y_pred.append(detected_lang)
                else:
                    logger.error("Evaluated URL language (lang: %s) not in the langs to be processed: this will falsify the evaluation: %s", gs_url2lang[url], url)
                    logger.error("This shouldn't be happening: bug?")
            else:
                logger.error("Evaluated URL not present in the GS: this will falsify the evaluation: %s", url)

        # Print results?
        if print_pairs and (print_negative_matches or detected_lang != _unknown_lang_label):
            if print_score:
                if gs_provided:
                    true_lang = _unknown_lang_label

                    if url in gs_url2lang:
                        true_lang = gs_url2lang[url]

                    print(f"{url}\t{detected_lang}\t{true_lang}")
                else:
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
            logger.warning("Unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 1, len(line))

            continue

        url = line[0]

        urls.append(url)

    logger.info("Provided URLs: %d", len(urls))

    # Evaluate
    logger.info("Evaluating...")

    y_pred, y_true, matches =\
        evaluate(urls, gs, gs_url2lang, gs_lang2url, lowercase=lowercase,
                 print_negative_matches=print_negative_matches, print_score=print_score)

    # Some statistics
    negative_matches = len(urls) - matches

    logger.info("Positive matches (i.e. language detected): %d", matches)
    logger.info("Negative matches (i.e. language not detected): %d", negative_matches)

    if gs_file:
        seen_langs = set.union(set(y_true), set(y_pred))
        seen_langs = sorted(list(seen_langs))

        logger.info("Using GS in order to get some evaluation metrics. Languages to process: %s", str(seen_langs))

        # Log metrics
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=_langs_to_detect)

        #logger.info("GS: confusion matrix: %s", list(confusion_matrix))

        precision = sklearn.metrics.precision_score(y_true, y_pred, labels=_langs_to_detect, average=None)
        recall = sklearn.metrics.recall_score(y_true, y_pred, labels=_langs_to_detect, average=None)
        f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=_langs_to_detect, average=None)

        for idx, lang in enumerate(_langs_to_detect):
            if lang not in seen_langs:
                continue

            incorrect = sum(list(confusion_matrix[idx])) - confusion_matrix[idx][idx]

            logger.info("GS: lang %s: confusion matrix row: %s (ok: %d; nok: %d)", lang, list(confusion_matrix[idx]), confusion_matrix[idx][idx], incorrect)
            logger.info("GS: lang %s: precision: %s", lang, precision[idx])
            logger.info("GS: lang %s: recall: %s", lang, recall[idx])
            logger.info("GS: lang %s: F1: %s", lang, f1[idx])

        for average in ("micro", "macro"):
            precision = sklearn.metrics.precision_score(y_true, y_pred, labels=_langs_to_detect, average=average)
            recall = sklearn.metrics.recall_score(y_true, y_pred, labels=_langs_to_detect, average=average)
            f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=_langs_to_detect, average=average)

            logger.info("GS: %s precision: %s", average, precision)
            logger.info("GS: %s recall: %s", average, recall)
            logger.info("GS: %s F1: %s", average, f1)

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

    logger = utils.set_up_logging_logger(logger, level=logging.DEBUG if args.verbose else logging.INFO)

    logger.debug("Arguments processed: {}".format(str(args)))

    main(args)
