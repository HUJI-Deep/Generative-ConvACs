#!/usr/bin/env python
import os, sys, glob, re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
def abspath_to_resource(path):
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))

find_p = re.compile(r"out_test_mar_(\d+\.\d+)\.txt")
find_acc = re.compile(r"[a-zA-Z ]*(\d+(\.\d+)?)")

def process_file(filename):
    try:
        with open(filename, 'r') as f:
            c = find_acc.match(f.readline())
            return float(c.group(1))
    except:
        return None

def main():
    results = []
    if os.path.isfile(abspath_to_resource('out_test_clean.txt')):
        acc = process_file(abspath_to_resource('out_test_clean.txt'))
        if acc is not None:
            results.append((0.0, acc))
    for f in glob.glob('%s/out_test_mar_*.txt' % (SCRIPT_DIR)):
        p = float(find_p.match(os.path.basename(f)).group(1))
        acc = process_file(f)
        if acc is not None:
            results.append((p, acc))
    if len(results) == 0:
        sys.exit(0)
    results = sorted(results)
    header, accuracy = zip(*results)
    print '==============' + '==='.join(['====']*len(header))
    print 'Probability | ' + ' | '.join(map(lambda x: '%.2f' % x, header))
    print '==============' + '==='.join(['====']*len(header))
    print 'Accuracy    | ' + ' | '.join(map(lambda x: '%.1f' % (x * 100), accuracy))
    print '==============' + '==='.join(['====']*len(header))

if __name__ == '__main__':
    main()
