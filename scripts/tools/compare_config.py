# Based on https://stackoverflow.com/questions/19120489/compare-two-files-report-difference-in-python
import sys
import argparse
from difflib import unified_diff

parser = argparse.ArgumentParser(description="Find differences between config files.")
parser.add_argument('--config1', type=str, help='configuration file (yaml)')
parser.add_argument('--config2', type=str, help='configuration file (yaml)')
args = parser.parse_args()

with open(args.config1) as f1:
    f1_text = f1.read().strip().splitlines()
with open(args.config2) as f2:
    f2_text = f2.read().strip().splitlines()

# Find and print the diff:
num_change = 0
for line in unified_diff(f1_text, f2_text, fromfile='config1', tofile='config2', lineterm='', n=0):
    for prefix in ('@@'):
        if line.startswith(prefix):
            break
    else:
        num_change += 1
        print(line)

print('\nTotal number of changes: %d' % num_change)