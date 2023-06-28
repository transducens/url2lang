
# Based on https://aclanthology.org/W16-2366.pdf 4.2

import os
import sys
import logging
import argparse

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import url2lang.url2lang as url2lang
import url2lang.utils.utils as utils

import sklearn.metrics
from tldextract import extract
import pycountry

# Disable (less verbose) 3rd party logging
logging.getLogger("filelock").setLevel(logging.WARNING)

# Order is relevant (because of greedy policy): same order as found in source
_langs_to_detect_alpha_3 = url2lang._langs_to_detect_alpha_3
_unknown_lang_label = url2lang._unknown_lang_label
_langs_to_detect = [_unknown_lang_label]

_3_letter_to_2_letter = False # Look for 2-letter code in URLs
_3_letter_to_2_letter_force = False # Do not add other thing which is not 2-letter code

def global_preprocessing():
    global _langs_to_detect

    initial_languages_skip = len(_langs_to_detect)

    # Get languages which will be detected in URLs
    for _lang in _langs_to_detect_alpha_3:
        if _lang == _unknown_lang_label:
            # Unknown data will not be taken into account
            continue

        _lang_to_add = None

        if _3_letter_to_2_letter:
            lang_alpha_2 = pycountry.languages.get(alpha_3=_lang)

            if "alpha_2" in dir(lang_alpha_2) and lang_alpha_2.alpha_2 is not None:
                _lang_to_add = lang_alpha_2.alpha_2
            else:
                if _3_letter_to_2_letter_force:
                    if lang_alpha_2 is None or "alpha_2" not in dir(lang_alpha_2) or lang_alpha_2.alpha_2 is None:
                        logging.error("Language %s couldn't be processed", _lang)
                    else:
                        logging.error("Language %s couldn't be processed: %s", lang_alpha_2, _lang)
                else:
                    logging.warning("Language %s: couldn't get 2-letter form: using initial value", _lang)

                    _lang_to_add = _lang
        else:
            _lang_to_add = _lang

        # Add lang
        if _lang_to_add is not None:
            if _lang_to_add not in _langs_to_detect:
                _langs_to_detect.append(_lang_to_add)
            else:
                logging.debug("Language %s already loaded: %s", _lang, _lang_to_add)

    logging.debug("%d languages loaded (initial languages: %d): %s",
                  len(_langs_to_detect) - initial_languages_skip, len(_langs_to_detect_alpha_3), ", ".join(_langs_to_detect))

_warning_once_done = False
def get_gs(file, ignore_unk=True):
    def unk_lang(lang):
        global _warning_once_done

        if lang == _unknown_lang_label:
            # Skip URLs whose lang is unknown in the GS

            if not _warning_once_done:
                _warning_once_done = True

                logging.warning("GS: URLs whose lang is %s are going to be ignored", lang)

            return True

        return False

    gs, url2lang, lang2url = set(), {}, {}

    for idx, line in enumerate(file, 1):
        line = line.rstrip("\r\n").split('\t')

        if len(line) != 2:
            logging.warning("GS: unexpected number of fields in TSV entry #%d: %d were expected but got %d", idx, 2, len(line))

            continue

        url, lang = line

        if ignore_unk and unk_lang(lang):
            continue

        if _3_letter_to_2_letter and len(lang) == 3:
            lang_alpha_2 = pycountry.languages.get(alpha_3=lang)

            if "alpha_2" in dir(lang_alpha_2) and lang_alpha_2.alpha_2 is not None:
                lang_alpha_2 = lang_alpha_2.alpha_2
                lang = lang_alpha_2
            # else: we don't need a warning: best effort approach

        if ignore_unk and unk_lang(lang):
            continue

        if lang not in _langs_to_detect:
            logging.warning("GS: URL language (lang: %s) not in the list of languages to be detected: entry #%d", lang, idx)

            continue

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

    global _warning_once_done

    _warning_once_done = False

    return gs, url2lang, lang2url

def main(args):
    input_file = args.input
    gs_file = args.gold_standard

    ifile, ifile_url2lang, ifile_lang2url = get_gs(input_file, ignore_unk=False)
    gs, gs_url2lang, gs_lang2url = get_gs(gs_file, ignore_unk=True)
    seen_urls = 0
    y_pred, y_true = [], []

    for ifile_url, ifile_lang in ifile_url2lang.items():
        if ifile_url not in gs_url2lang.keys():
            logging.warning("URL not in GS: %s", ifile_url)

            continue

        if gs_url2lang[ifile_url] in _langs_to_detect:
            y_true.append(gs_url2lang[ifile_url])
            y_pred.append(ifile_url2lang[ifile_url])
        else:
            logging.error("Evaluated URL language (lang: %s) not in the langs to be processed: this will falsify the evaluation: %s", gs_url2lang[url], url)
            logging.error("This shouldn't be happening: bug?")

        seen_urls += 1

    logging.debug("In common with GS, total and GS elements: %d %d %d", seen_urls, len(ifile_url2lang.keys()), len(gs_url2lang.keys()))

    if seen_urls < len(gs_url2lang.keys()):
        logging.warning("%d elements in GS, but %d are missing", seen_urls, len(gs_url2lang.keys()) - seen_urls)

    # Evaluate
    seen_langs = set.union(set(y_true), set(y_pred))
    seen_langs = sorted(list(seen_langs))

    logging.info("Using GS in order to get some evaluation metrics. Languages to process: %s", str(seen_langs))

    # Log metrics
    labels = sorted(list(set(y_true)))
    labels_conf_mat = _langs_to_detect
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels_conf_mat)

    #logging.info("GS: confusion matrix: %s", list(confusion_matrix))

    precision = sklearn.metrics.precision_score(y_true, y_pred, labels=labels, average=None)
    recall = sklearn.metrics.recall_score(y_true, y_pred, labels=labels, average=None)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average=None)

    for idx, lang in enumerate(labels):
        if lang not in _langs_to_detect:
            logging.warning("Couldn't process lang %s: is not in the list of available langs", lang)

            continue

        conf_mat_idx = labels_conf_mat.index(lang)
        # Multiclass TP, FN, FP and TN (https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/)
        tp = confusion_matrix[conf_mat_idx][conf_mat_idx]
        fn = sum(list(confusion_matrix[conf_mat_idx])) - tp
        fp = sum(list([confusion_matrix[i][conf_mat_idx] for i in range(len(confusion_matrix))])) - tp
        tn = confusion_matrix.sum() - fn - fp - tp

        logging.info("GS: lang %s: confusion matrix row: [%s]", lang, ", ".join([f"{_lang}: {cm}" for _lang, cm in zip(labels_conf_mat, list(confusion_matrix[conf_mat_idx]))]))
        logging.info("GS: lang %s: TP, FP, TN, FN: %d %d %d %d", lang, tp, fp, tn, fn)
        logging.info("GS: lang %s: precision: %s", lang, precision[idx])
        logging.info("GS: lang %s: recall: %s", lang, recall[idx])
        logging.info("GS: lang %s: F1: %s", lang, f1[idx])

    for average in ("micro", "macro"):
        precision = sklearn.metrics.precision_score(y_true, y_pred, labels=labels, average=average)
        recall = sklearn.metrics.recall_score(y_true, y_pred, labels=labels, average=average)
        f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average=average)

        logging.info("GS: %s precision (%d classes): %s", average, len(labels), precision)
        logging.info("GS: %s recall (%d classes): %s", average, len(labels), recall)
        logging.info("GS: %s F1 (%d classes): %s", average, len(labels), f1)

    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)

    logging.info("GS: Accuracy: %s", acc)
    logging.info("GS: MCC: %s", mcc)

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Get metrics")

    parser.add_argument('input', type=argparse.FileType('rt', errors="backslashreplace"), help="Filename with URLs and lang (TSV format)")
    parser.add_argument('gold_standard', type=argparse.FileType('rt', errors="backslashreplace"), help="GS filename with URLs and lang (TSV format)")

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
