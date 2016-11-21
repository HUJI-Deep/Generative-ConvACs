#!/usr/bin/env python
import os, subprocess, time, signal


from os.path import join

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_SCRIPT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../utils/knn_missing_data.py'))
enclosed_dir = os.path.normpath(os.path.join(SCRIPT_DIR, '../'))
subprocess.check_call('%s/generate_norb_small_missing_data.py' % enclosed_dir, shell=True)

NORB_DATA = '../data/'
NORB_TEST = '../data/norb_small_2D_test_image_data'
MAR_PROBS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
RECT_NUMS = [1, 2, 3]
RECT_WIDTHS = [7, 11, 15]
DBS = [NORB_TEST]
K = 5

def test(name, training_data, dataset, K):
    if os.path.isfile(name):
        return
    else:
        touch(name)
    cmd = [TEST_SCRIPT, '--data_dir', join(SCRIPT_DIR,NORB_DATA), '--missing_data_dir', 
           join(SCRIPT_DIR,dataset), '--k', str(K), '--dataset', 'norb']
    cmd = ' '.join(cmd)
    print cmd
    results = subprocess.check_output(cmd, shell=True, cwd=SCRIPT_DIR)
    with open(name, 'wb') as f:
        f.write(results)

try:
    for db in DBS:
        for mar_prob in MAR_PROBS:
            base = 'mar_%.2f' % (mar_prob)
            dataset = '%s_%s' % (db, base)
            name = 'out_test_%s.csv' % (base)
            test(name, NORB_DATA, dataset, K)
        for rect_num in RECT_NUMS:
            for rect_width in RECT_WIDTHS:
                base = 'minrects_{0}_maxrects_{0}_minwidth_{1}_maxwidth_{1}'.format(rect_num, rect_width)
                dataset = '%s_%s' % (db, base)
                name = 'out_test_%s.csv' % (base)
                test(name, NORB_DATA, dataset, K)
except KeyboardInterrupt:
    time.sleep(1)
    print 'Error during testing.'