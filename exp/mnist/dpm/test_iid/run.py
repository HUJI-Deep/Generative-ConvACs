#!/usr/bin/env python
import os, subprocess, signal, sys

train_dir = os.path.dirname(os.path.realpath(__file__))
enclosed_dir = os.path.normpath(os.path.join(train_dir, '../../'))
caffe_dir = os.path.abspath(os.path.join(train_dir, '../../../../deps/simnets'))
model_path = os.path.join(enclosed_dir, 'lenet/train/lenet_train_1_iter_10000.caffemodel')
if not os.path.isfile(model_path):
    print 'Cannot find pretrained model at %s. You must either train the model first or download a readymade model.' % (model_path)
    sys.exit(-1)
subprocess.check_call('%s/generate_mnist_missing_data.py' % enclosed_dir, shell=True)
try:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    subprocess.check_call('%s/tools/extra/hyper_train.py %s/net.prototmp %s/train_plan.json' % (caffe_dir, train_dir,train_dir), shell=True, cwd=enclosed_dir)
except:
    print 'Error calling hyper_train script'

