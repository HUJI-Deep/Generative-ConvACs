#!/usr/bin/env python
import os, subprocess, signal, time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
GEN_SCRIPT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../utils/generate_missing_data_dataset.py'))
MNIST_TEST = 'data/mnist_test_image_data'
MNIST_TRAIN = 'data/mnist_train_image_data'
MNIST_VALID = 'data/mnist_validation_image_data'
MNIST_WHOLE_TRAIN = 'data/mnist_whole_train_image_data'
DBS = [MNIST_TRAIN, MNIST_WHOLE_TRAIN, MNIST_VALID, MNIST_TEST]
MAR_PROBS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
RECT_NUMS = [1, 2, 3]
RECT_WIDTHS = [8, 12, 16]

subprocess.check_call('%s/generate_mnist.py' % SCRIPT_DIR, shell=True)
try:
    for db in DBS:
        for mar_prob in MAR_PROBS:
            cmd = '%s --MAR_prob %.2f %s ' % (GEN_SCRIPT, mar_prob, db)
            print cmd
            subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR)
        for rect_num in RECT_NUMS:
            for rect_width in RECT_WIDTHS:
                cmd = '{0} --min_rects {1} --max_rects {1} --min_width {2} --max_width {2} {3} '.format(GEN_SCRIPT, rect_num, rect_width, db)
                print cmd
                subprocess.check_call(cmd, shell=True, cwd=SCRIPT_DIR)
except KeyboardInterrupt:
    time.sleep(1)
    print 'Error while generating missing data datasets!'