#!/usr/bin/env python
import os, subprocess, signal

train_dir = os.path.dirname(os.path.realpath(__file__))
enclosed_dir = os.path.normpath(os.path.join(train_dir, '../../'))
caffe_dir = os.path.abspath(os.path.join(train_dir, '../../../../deps/simnets'))
subprocess.check_call('%s/generate_norb_small.py --aug_data y' % enclosed_dir, shell=True)

try:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if not os.path.isfile(os.path.join(train_dir, 'init/ght_model_train_init_1_iter_1.caffemodel')):
        print 'Generating pre-trained model:'
        print ''
        subprocess.check_call('%s/init/run.py' % train_dir, shell=True)
        print '=============== DONE ==============='
    for loss_weight_dir in ['loss_weight_0.01', 'loss_weight_0.1']:
        cmd = '{0}/tools/extra/hyper_train.py {1}/{2}/net.prototmp {1}/{2}/train_plan.json --gpu all'.format(caffe_dir, train_dir, loss_weight_dir)
        subprocess.check_call(cmd, shell=True, cwd=enclosed_dir)
except:
    print 'Error calling hyper_train script'
