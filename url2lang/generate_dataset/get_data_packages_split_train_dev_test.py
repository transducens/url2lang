
import os
import sys
import random
import logging

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f"{cdir}/../..")

import url2lang.utils.utils as utils

from tldextract import extract

utils.set_up_logging(level=logging.DEBUG)

logging.getLogger("filelock").setLevel(logging.WARNING)

random.seed(42)

n_packages = int(sys.argv[1])
packages_quantity = int(sys.argv[2])
domain_train_perc = float(sys.argv[3])
domain_dev_perc = float(sys.argv[4])
domain_test_perc = float(sys.argv[5])

sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

# n_urls_per_package = int(packages_quantity / n_packages + 0.5)
max_urls_per_lang = n_packages * packages_quantity
domain2url_and_lang = {}
domain2lang2quantity = {}
lang2domain2quantity = {}
lang2quantity = {}

logging.info("Loading data")

for l in sys.stdin:
    l = l.rstrip("\r\n").split('\t')
    url, lang = l

    if len(url) > 2000:
        continue

    if lang in lang2quantity and lang2quantity[lang] >= max_urls_per_lang:
        continue

    if lang not in lang2quantity:
        lang2quantity[lang] = 0

    domain = extract(url)[1]

    if domain not in domain2url_and_lang:
        domain2url_and_lang[domain] = []

    if domain not in domain2lang2quantity:
        domain2lang2quantity[domain] = {}

    if lang not in lang2domain2quantity:
        lang2domain2quantity[lang] = {}

    if domain not in lang2domain2quantity[lang]:
        lang2domain2quantity[lang][domain] = 0

    if lang not in domain2lang2quantity[domain]:
        domain2lang2quantity[domain][lang] = 0

    domain2url_and_lang[domain].append((url, lang))
    domain2lang2quantity[domain][lang] += 1
    lang2domain2quantity[lang][domain] += 1
    lang2quantity[lang] += 1

logging.info("Shuffling data")

# Shuffle data
for domain in sorted(list(domain2url_and_lang.keys())):
    random.shuffle(domain2url_and_lang[domain])

total_domains = len(domain2lang2quantity.keys())
dev_n_domains = int(domain_dev_perc * total_domains + 0.5)
test_n_domains = int(domain_test_perc * total_domains + 0.5)
train_n_domains = total_domains - dev_n_domains - test_n_domains

if train_n_domains <= 0:
    logging.warning("Train n domains <= 0: %d", train_n_domains)

    train_n_domains = test_n_domains = int(domain_train_perc * total_domains + 0.5)
    test_n_domains = total_domains - train_n_domains - dev_n_domains

logging.info("N domains train, dev, test: %d %d %d", train_n_domains, dev_n_domains, test_n_domains)

seen_domains = set()
all_domains = list(domain2lang2quantity.keys())
all_domains = random.sample(all_domains, k=len(all_domains))
all_langs = list(lang2domain2quantity.keys())
all_langs = random.sample(all_langs, k=len(all_langs))

logging.info("Processing data")

# Dev, test and, finally, train
print("set\tpackage\turl\tlang") # Header

for s, set_n_domains, total_packages in (("dev", dev_n_domains, 1), ("test", test_n_domains, 1), ("train", train_n_domains, n_packages)):
    max_urls_per_lang_per_package = int(packages_quantity / len(all_langs) / total_packages + 0.5)
    max_domains_per_lang_per_package = int(set_n_domains / len(all_langs) / total_packages + 0.5)

    for n_package in range(total_packages):
        added_domains = {l: 0 for l in all_langs}
        quantity_urls = {l: 0 for l in all_langs}

        for domain in all_domains:
            if domain in seen_domains:
                continue

            if added_domains[lang] >= max_domains_per_lang_per_package:
                break

            c = False

            for lang in domain2lang2quantity[domain]:
                # Check if we can process all the languages of the current domain
                if quantity_urls[lang] + domain2lang2quantity[domain][lang] >= max_urls_per_lang_per_package:
                    # The domain would add more pairs than allowed per lang for the current lang
                    c = True

                    break

            if c:
                continue

            debug_check = {l: 0 for l in all_langs}

            for url, lang in domain2url_and_lang[domain]:
                print(f"{s}\t{n_package}\t{url}\t{lang}")

                debug_check[lang] += 1

            # Update control vars
            seen_domains.add(domain)

            for lang in domain2lang2quantity[domain]:
                quantity_urls[lang] += domain2lang2quantity[domain][lang]
                added_domains[lang] += 1

                if debug_check[lang] != domain2lang2quantity[domain][lang]:
                    logging.error("Bug: check debug_check var")

        for lang in all_langs:
            logging.debug("%s: package %d: lang %s: difference in max domains per package per lang and processed domains: %d - %d = %d",
                          s, n_package, lang, max_domains_per_lang_per_package, added_domains[lang], max_domains_per_lang_per_package - added_domains[lang])
            logging.debug("%s: package %d: lang %s: difference in max URLs per package per lang and processed URLs: %d - %d = %d",
                          s, n_package, lang, max_urls_per_lang_per_package, quantity_urls[lang], max_urls_per_lang_per_package - quantity_urls[lang])

# Are there domains which were not added? Add to dev test
for remaining_domain in set.difference(set(all_domains), seen_domains):
    for url, lang in domain2url_and_lang[domain]:
        print(f"dev\t-1\t{url}\t{lang}") # package -1 in order to be able to identify this "extra" data

logging.info("Done!")
