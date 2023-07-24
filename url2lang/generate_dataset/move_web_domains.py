
import sys

from tldextract import extract

sys.stdin.reconfigure(encoding='utf-8', errors="backslashreplace")

filename1 = sys.argv[1]
filename2 = sys.argv[2]
direction = sys.argv[3]
output_filename_suffix = sys.argv[4]

if direction not in ("first2second", "second2first"):
    raise Exception("3rd param syntax: first2second | second2first")

domains = set()

for domain in sys.stdin:
    domains.add(domain.rstrip("\r\n"))

def read_file(filename):
    domain2urls = {}

    with open(filename) as fd:
        for l in fd:
            data = extract(l.rstrip("\r\n").split('\t')[0]) # URL is expected to be in the first column

            if data[1]:
                domain = data[1]
            else:
                domain = data[2]

            if domain not in domain2urls:
                domain2urls[domain] = []

            domain2urls[domain].append(l)

    return domain2urls

data1 = read_file(filename1)
data2 = read_file(filename2)
src = data1 # first2second direction
trg = data2

if direction == "second2first":
    trg = data1
    src = data2

for domain in domains:
    if domain not in src:
        sys.stderr.write(f"WARNING: domain '{domain}' not found in source data\n")

        continue

    if domain not in trg:
        trg[domain] = []

    trg[domain].extend(src[domain])

    del src[domain] # data now is only in trg

def write_file(output_filename_prefix, data):
    filename = f"{output_filename_prefix}.{output_filename_suffix}"

    with open(filename, 'w') as fd:
        for urls in data.values():
            for url in urls:
                fd.write(url)

# Write
write_file(filename1, data1)
write_file(filename2, data2)
