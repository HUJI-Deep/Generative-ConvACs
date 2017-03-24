#!/usr/bin/env python
import os, subprocess, signal

train_dir = os.path.dirname(os.path.realpath(__file__))
enclosed_dir = os.path.normpath(os.path.join(train_dir, '../../'))
caffe_dir = os.path.abspath(os.path.join(train_dir, '../../../../deps/simnets'))
subprocess.check_call('%s/generate_mnist.py' % enclosed_dir, shell=True)

try:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    subprocess.check_call('%s/tools/extra/hyper_train.py %s/net.prototmp %s/train_plan.json --gpu all' % (caffe_dir, train_dir,train_dir), shell=True, cwd=enclosed_dir)
except:
    print 'Error calling hyper_train script'

