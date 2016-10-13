#!/usr/bin/env python
import os, subprocess, signal

train_dir = os.path.dirname(os.path.realpath(__file__))
enclosed_dir = os.path.normpath(os.path.join(train_dir, '../../'))
caffe_dir = os.path.abspath(os.path.join(train_dir, '../../../../deps/simnets'))
subprocess.check_call('%s/generate_mnist.py' % enclosed_dir, shell=True)

try:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if not os.path.isfile(os.path.join(train_dir, 'init/ht_model_train_init_1_iter_1.caffemodel')):
        print 'Generating pre-trained model:'
        print ''
        subprocess.check_call('%s/init/run.py' % train_dir, shell=True)
        print '=============== DONE ==============='
    subprocess.check_call('%s/tools/extra/hyper_train.py %s/net.prototmp %s/train_plan.json --gpu all' % (caffe_dir, train_dir,train_dir), shell=True, cwd=enclosed_dir)
except:
    print 'Error calling hyper_train script'

