
import os
import sys
import random
import logging
import argparse
import urllib.parse
import contextlib

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import url2lang.utils.utils as utils
import url2lang.preprocess as preprocess

import numpy as np
from tldextract import extract

# Disable (less verbose) 3rd party logging
logging.getLogger("filelock").setLevel(logging.WARNING)

def store_negative_samples(parallel_urls, non_parallel_filename, target_domains, unary_generator, logging_cte=2):
    no_parallel_domains = len(target_domains)
    no_non_parallel_urls = 0
    no_non_parallel_domains = 0
    last_perc_shown = -1

    with open(non_parallel_filename, 'w') as f:
        for idx, domain in enumerate(target_domains):
            parallel_urls_domain = list(parallel_urls[domain]) # WARNING: you will need to set PYTHONHASHSEED if you want deterministic results across different executions
            negative_samples = unary_generator(parallel_urls_domain)

            for src_url, trg_url in negative_samples:
                f.write(f"{src_url}\t{trg_url}\n")

            _finished_perc = (idx + 1) * 100.0 / no_parallel_domains
            finished_perc = int(_finished_perc)

            # Update statistics
            no_non_parallel_domains += 1 if len(negative_samples) != 0 else 0
            no_non_parallel_urls += len(negative_samples)

            # Show every logging_cte %
            if finished_perc % logging_cte == 0 and finished_perc != last_perc_shown:
                logging.info("%.2f %% of negative samples generated (%d out of %d domains were already processed): %d negative samples",
                             _finished_perc, idx + 1, no_parallel_domains, no_non_parallel_urls)

                last_perc_shown = finished_perc

    return no_non_parallel_urls, no_non_parallel_domains

def store_dataset(parallel_urls, target_domains, parallel_filename, non_parallel_filename, logging_cte=2,
                  process_even_if_files_exist=False):
    no_parallel_urls = 0
    no_parallel_domains = len(target_domains)
    last_perc_shown = -1
    write_positive_samples = process_even_if_files_exist or not utils.exists(parallel_filename)
    positive_samples_cm = open(parallel_filename, 'w') if write_positive_samples else contextlib.nullcontext()

    if not write_positive_samples:
        logging.warning("Positive samples will not be generated since already exist, but statistics will be displayed: %s", parallel_filename)

    # Store parallel URLs
    with positive_samples_cm as f:
        for idx, domain in enumerate(target_domains):
            for url1, url2 in parallel_urls[domain]:
                f.write(f"{url1}\t{url2}\n")

                no_parallel_urls += 1

            _finished_perc = (idx + 1) * 100.0 / no_parallel_domains
            finished_perc = int(_finished_perc)

            # Show every logging_cte %
            if finished_perc % logging_cte == 0 and finished_perc != last_perc_shown:
                logging.info("%.2f %% of positive samples generated (%d out of %d domains were already processed): %d positive samples",
                             _finished_perc, idx + 1, no_parallel_domains, no_parallel_urls)

                last_perc_shown = finished_perc

    logging.info("Total URLs (positive samples): stored in '%s': %d", parallel_filename, no_parallel_urls)
    logging.info("Total domains (positive samples): %d", no_parallel_domains)

def main(args):
    input_file_parallel_urls = args.input_file_parallel_urls
    output_file_urls_prefix = args.output_files_prefix
    seed = args.seed
    check_same = args.check_same
    sets_absolute_instead_of_relative = args.sets_absolute_instead_of_relative
    sets_percentage = args.sets_percentage
    process_even_if_files_exist = args.process_even_if_files_exist

    train_perc, dev_perc, test_perc = sets_percentage
    sets_quantities_with_minus_one = -1

    if sets_absolute_instead_of_relative:
        sets_quantities_with_minus_one = sets_percentage.count(-1)

        if sets_quantities_with_minus_one > 1:
            raise Exception("The provided sets quantities have the value -1 >1 times and only it is allowed [0,1] times")
    elif not np.isclose(sum(sets_percentage), 1.0):
        raise Exception("The provided sets percentages do not sum up to 1.0")

    if "PYTHONHASHSEED" not in os.environ:
        logging.warning("You did not provide PYTHONHASHSEED: the results will not be deterministic")

    if seed >= 0:
        random.seed(seed)

    parallel_urls = {}
    skipped_urls = 0
    no_parallel_urls = 0

    for idx, url_pair in enumerate(input_file_parallel_urls, 1):
        url_pair = url_pair.strip().split('\t')

        if len(url_pair) != 2:
            raise Exception(f"The provided line does not have 2 tab-separated values but {len(url_pair)} (line #{idx})")

        if len(url_pair[0]) == 0 or len(url_pair[1]) == 0:
            logging.warning("Skipping line #%d because there are empty values", idx)
            skipped_urls += 1

            continue
        if len(url_pair[0]) > 1000 or len(url_pair[1]) > 1000:
            logging.warning("Skipping line #%d because there are URLs too long (%d and %d)", idx, len(url_pair[0]), len(url_pair[1]))
            skipped_urls += 1

            continue

        url_pair = preprocess.preprocess_url(url_pair, separator='/', remove_protocol=False, tokenization=False)
        src_subdomain, src_domain, src_tld = extract(url_pair[0])
        trg_subdomain, trg_domain, trg_tld = extract(url_pair[1])
        domains = (src_domain, trg_domain) # We are grouping by domain
        src_check, trg_check = '', ''

        if check_same == "authority":
            src_check = f"{src_subdomain}.{src_domain}.{src_tld}"
            trg_check = f"{trg_subdomain}.{trg_domain}.{trg_tld}"
        elif check_same == "subdomain":
            src_check = src_subdomain
            trg_check = trg_subdomain
        elif check_same == "domain":
            src_check = src_domain
            trg_check = trg_domain
        elif check_same == "tld":
            src_check = src_tld
            trg_check = trg_tld
        elif check_same == "none":
            pass
        else:
            raise Exception(f"Unknown 'check_same' option: {check_same}")

        if src_check != trg_check:
            logging.debug("Skipping line #%d because the URLs didn't pass the check (%s vs %s)", idx, src_check, trg_check)
            skipped_urls += 1

            continue

        domain = f"{domains[0]}\t{domains[1]}"

        if domain not in parallel_urls:
            parallel_urls[domain] = set()

        _no_parallel_urls_before = len(parallel_urls[domain])
        parallel_urls[domain].add((url_pair[0], url_pair[1]))
        _no_parallel_urls_after = len(parallel_urls[domain])

        no_parallel_urls += 1 if _no_parallel_urls_before != _no_parallel_urls_after else 0

    len_domains = len(parallel_urls.keys())
    total_read_urls = idx

    logging.info("Skipped lines: %d out of %d (%.2f %%) due to checks; %d out of %d (%.2f %%) due to duplicates",
                 skipped_urls, total_read_urls, skipped_urls * 100.0 / total_read_urls,
                 no_parallel_urls, total_read_urls, no_parallel_urls * 100.0 / total_read_urls)
    logging.info("Loaded URLs (positive samples): %d", no_parallel_urls)
    logging.info("Total domains (positive samples): %d", len_domains)

    if sets_absolute_instead_of_relative:
        if sets_quantities_with_minus_one == 1:
            # Update the set quantity
            for i, v in enumerate(sets_percentage):
                if v == -1:
                    new_value = len_domains - (sum(sets_percentage) - sets_percentage[i])
                    label = "train" if i == 0 else "dev" if i == 1 else "test" if i == 2 else "UNKNOWN"

                    if new_value < 0:
                        raise Exception(f"Not enough web domains ({len_domains}): you set {sets_percentage} and "
                                        f"{new_value} ({label} set) < 0 but should be >=")

                    sets_percentage[i] = new_value

            train_perc, dev_perc, test_perc = sets_percentage

        sum_sets = sum(sets_percentage)

        if sum_sets != len_domains:
            raise Exception(f"The provided sets quantities do not sum up to the total number of domains: {sets_percentage} -> {sum_sets} != {len_domains}")

        train_max_idx = 0 + train_perc
        dev_max_idx = train_max_idx + dev_perc
        test_max_idx = dev_max_idx + test_perc
    else:
        train_max_idx = 0 + int(train_perc * len_domains)
        dev_max_idx = train_max_idx + int(dev_perc * len_domains)
        test_max_idx = len_domains # test_perc is not used since it is sure that the percentages sum up to 1.0 and this method might
                                   #  leave some domain out

        if np.isclose(test_perc, 0.0) and test_max_idx > dev_max_idx:
            # Special behaviour needed due to precision problems
            logging.warning("Since test percentage was set to 0 but some domains were added, they are going to be used in the dev set instead")

            dev_max_idx += test_max_idx - dev_max_idx
            test_max_idx = dev_max_idx

    train_domains = set()
    dev_domains = set()
    test_domains = set()
    all_domains = list(parallel_urls.keys())

    random.shuffle(all_domains) # Shuffle domains in order to avoid dependency between the data and the order it was provided

    for idx, domain in enumerate(all_domains):
        if idx < train_max_idx:
            train_domains.add(domain)
        elif idx < dev_max_idx:
            dev_domains.add(domain)
        elif idx < test_max_idx:
            test_domains.add(domain)
        else:
            logging.error("Domain is not going to be processed in iteration %d: %s", idx, domain)

    logging.info("Train domains: %d", len(train_domains))
    logging.info("Dev domains: %d", len(dev_domains))
    logging.info("Test domains: %d", len(test_domains))

    if len(train_domains) + len(dev_domains) + len(test_domains) != len_domains:
        raise Exception("Not all the domains have been set to a set")

    # Generate positive and negative samples for train, dev and test sets
    store_dataset(parallel_urls, train_domains,
                  f"{output_file_urls_prefix}.parallel.train", f"{output_file_urls_prefix}.non-parallel.train",
                  logging_cte=5, process_even_if_files_exist=process_even_if_files_exist)
    store_dataset(parallel_urls, dev_domains,
                  f"{output_file_urls_prefix}.parallel.dev", f"{output_file_urls_prefix}.non-parallel.dev",
                  logging_cte=10, process_even_if_files_exist=process_even_if_files_exist)
    store_dataset(parallel_urls, test_domains,
                  f"{output_file_urls_prefix}.parallel.test", f"{output_file_urls_prefix}.non-parallel.test",
                  logging_cte=10, process_even_if_files_exist=process_even_if_files_exist)

    logging.info("Done!")

def initialization():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Create URLs dataset from parallel samples")

    sets_absolute_instead_of_relative = "--sets-absolute-instead-of-relative" in sys.argv[1:]

    parser.add_argument('input_file_parallel_urls', type=argparse.FileType('rt', errors="backslashreplace"), help="Input TSV file with parallel URLs")
    parser.add_argument('output_files_prefix', help="Output files prefix")

    parser.add_argument('--check-same', choices=["none", "authority", "subdomain", "domain", "tld"], default="domain", help="Skip pair of URLs according to the specified configuration")
    parser.add_argument('--sets-percentage', type=int if sets_absolute_instead_of_relative else float, nargs=3, default=None if sets_absolute_instead_of_relative else [0.8, 0.1, 0.1],
                        required=sets_absolute_instead_of_relative,
                        help="Train, dev and test percentages (by default, relative percentages, but if --sets-absolute-instead-of-relative is set, absolute quantities are expected)")
    parser.add_argument('--process-even-if-files-exist', action='store_true', help="When output files exist, they are not generated. If set, they will be generated even if the files exist")
    parser.add_argument('--sets-absolute-instead-of-relative', action='store_true',
                        help="--sets-percentage option will accept an absolute number of web domains instead of a percentage of them. A value of -1 will be accepted in one or zero "
                             "positions and it will mean to inference the number of web domains")

    parser.add_argument('--force-non-deterministic', action='store_true', help="If PYTHONHASHSEED is not set, it will be set in order to obtain deterministic results. If this flag is set, this action will not be done")
    parser.add_argument('--seed', type=int, default=71213, help="Seed in order to have deterministic results (fully guaranteed if you also set PYTHONHASHSEED envvar). Set a negative number in order to disable this feature")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logging mode")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # https://stackoverflow.com/questions/16549332/python-3-how-to-specify-stdin-encoding
    sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

    args = initialization()

    utils.set_up_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if "PYTHONHASHSEED" not in os.environ and not args.force_non_deterministic:
        # TODO it closes opened temporary named pipe files -> the execution crashes -> how to fix and still keep this behavior?
        # Wrapper call in order to define PYTHONHASHSEED (https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program)

        PYTHONHASHSEED_value = args.seed

        logging.warning("PYTHONHASHSEED not set: using seed: %d", args.seed)

        import subprocess

        subprocess.run([sys.executable] + sys.argv, env={**dict(os.environ), **{"PYTHONHASHSEED": str(PYTHONHASHSEED_value)}})
    else:
        logging.debug("Arguments processed: {}".format(str(args)))

        if "PYTHONHASHSEED" in os.environ:
            logging.debug("PYTHONHASHSEED: %s", os.environ["PYTHONHASHSEED"])

        main(args)
