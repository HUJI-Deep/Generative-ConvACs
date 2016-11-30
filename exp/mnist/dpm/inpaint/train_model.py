#!/usr/bin/env python
import os, subprocess, time, sys
from os.path import join

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DPM_REPO_PATH = os.path.abspath(join(SCRIPT_DIR, '..', '..', '..', '..', 
										'deps', 'DPM'))
enclosed_dir = os.path.normpath(os.path.join(SCRIPT_DIR, '../../'))
subprocess.check_call('%s/generate_mnist.py' % 
                        enclosed_dir, shell=True)

TRAIN_SCRIPT = join(DPM_REPO_PATH, 'train.py')

cmd = [TRAIN_SCRIPT]
cmd = ' '.join(cmd)
print cmd
subprocess.check_call(cmd, shell=True, cwd=DPM_REPO_PATH)