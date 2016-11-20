#!/usr/bin/env python
import os, subprocess, time, sys
from os.path import join

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
enclosed_dir = os.path.normpath(os.path.join(SCRIPT_DIR, '../../'))
subprocess.check_call('%s/generate_mnist_missing_data.py' % 
                        enclosed_dir, shell=True)

# Trained DPM model location (model.pkl). To train it, follow instructions at 
# https://github.com/HUJI-Deep/Diffusion-Probabilistic-Models
model_path = 'path/to/trained/model.pkl'
dpm_repo_path = 'path/to/dpm/repo'



inpainting_script = os.path.abspath(os.path.join(SCRIPT_DIR, dpm_repo_path,
                                                 'inpaint_from_model.py'))




MNIST_DATA = '../../data/'
MNIST_TEST = join(MNIST_DATA, 'mnist_test_image_data')
IP_SUFFIX = '_dpm_ip'
MAR_PROBS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
RECT_NUMS = [1, 2, 3]
RECT_WIDTHS = [7, 11, 15]
DBS = [MNIST_TEST]


def inpaint_dataset(model_path, missing_data_path):
    model_norm_path = os.path.normpath(join(SCRIPT_DIR, model_path))
    missing_data_norm_path = os.path.normpath(join(SCRIPT_DIR, missing_data_path))
    
    if not os.path.isfile(model_norm_path):
        print ('Cannot find pretrained model at %s. You must either train the model'
        'first or download a readymade model.' % (model_norm_path))
        sys.exit(-1)
    if not os.path.exists(missing_data_norm_path):
        print ("%s doesn't exist, skipping to next dataset..." % 
                (missing_data_norm_path))
    if os.path.exists(missing_data_norm_path + IP_SUFFIX):
        print ("%s already exists, skipping to next dataset..." % 
                (missing_data_norm_path + IP_SUFFIX))
        return
    
    
    cmd = [inpainting_script, '--resume_file', model_norm_path, 
    '--missing_dataset_path', missing_data_norm_path]
    cmd = ' '.join(cmd)
    print cmd
    results = subprocess.check_output(cmd, shell=True, cwd=SCRIPT_DIR)
    return
    
     

try:
    for db in DBS:
        for mar_prob in MAR_PROBS:
            base = 'mar_%.2f' % (mar_prob)
            dataset = '%s_%s' % (db, base)
            name = 'out_test_%s.csv' % (base)
            inpaint_dataset(model_path, dataset)
        for rect_num in RECT_NUMS:
            for rect_width in RECT_WIDTHS:
                base = 'minrects_{0}_maxrects_{0}_minwidth_{1}_maxwidth_{1}'.format(rect_num, rect_width)
                dataset = '%s_%s' % (db, base)
                name = 'out_test_%s.csv' % (base)
                inpaint_dataset(model_path, dataset)
except KeyboardInterrupt:
    time.sleep(1)
    print 'Error during testing.'