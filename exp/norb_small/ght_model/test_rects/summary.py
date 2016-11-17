#!/usr/bin/env python
import os, sys, glob, re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

find_NW = re.compile(r"out_test_minrects_(\d+)_maxrects_(\d+)_minwidth_(\d+)_maxwidth_(\d+)\.txt")
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
    for f in glob.glob('%s/out_test_minrects_*_maxrects_*_minwidth_*_maxwidth_*.txt' % (SCRIPT_DIR)):
        m = find_NW.match(os.path.basename(f))
        N = int(m.group(1))
        W = int(m.group(3))
        acc = process_file(f)
        if acc is not None:
            results.append(((N,W), acc))
    if len(results) == 0:
        sys.exit(0)
    results = sorted(results, key=lambda x: (x[0][1], x[0][0]))
    header, accuracy = zip(*results)
    print '==============' + '==='.join(['======']*len(header))
    print '(N, W)      | ' + ' | '.join(map(lambda x: '(%d,%2d)' % x, header))
    print '==============' + '==='.join(['======']*len(header))
    print 'Accuracy    | ' + ' | '.join(map(lambda x: ' %.1f ' % (x * 100), accuracy))
    print '==============' + '==='.join(['======']*len(header))

if __name__ == '__main__':
    main()
