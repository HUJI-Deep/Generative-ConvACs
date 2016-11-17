#!/usr/bin/env python
import os, subprocess, signal, time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
def abspath_to_resource(path):
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))

TEST_SCRIPT = abspath_to_resource('../../../../utils/ensemble.py')
MNIST_TEST = abspath_to_resource('../../data/norb_small_2D_test_image_data')
DBS = [MNIST_TEST]
MODELS_PROTOS = map(lambda x: abspath_to_resource(x), [
    '../ght_model_deploy.prototxt'])
MODELS_WEIGHTS = map(lambda x: abspath_to_resource(x), [
    '../train/loss_weight_0.01/ght_model_train_loss_weight_0.01_1_iter_10000.caffemodel',
    '../train/loss_weight_0.01/ght_model_train_loss_weight_0.01_2_iter_10000.caffemodel',
    '../train/loss_weight_0.1/ght_model_train_loss_weight_0.1_1_iter_10000.caffemodel',
    '../train/loss_weight_0.1/ght_model_train_loss_weight_0.1_2_iter_10000.caffemodel'])
SCORE_TYPE = 'avg' # use vote for voting ensembles
MAR_PROBS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def test(name, dataset, models_protos, models_weights, missing_data=True, whitened_similarity=True):
    fname = abspath_to_resource(name)
    if os.path.isfile(fname):
        return
    else:
        touch(fname)
    cmd = [TEST_SCRIPT, '-is_color', 'y', '-single_accs', 'y', '-score_type', SCORE_TYPE,
          '-starts_with_whitened_similarity', 'y' if whitened_similarity else 'n']
    if missing_data:
        cmd += ['-missing_data_index_file', '%s/index_mask.txt' % dataset]
    cmd += [','.join(models_protos), ','.join(models_weights), '%s/index.txt' % dataset, 'prob']
    cmd = ' '.join(cmd)
    print cmd
    results = subprocess.check_output(cmd, shell=True, cwd=SCRIPT_DIR)
    with open(fname, 'wb') as f:
        f.write(results)

for model in MODELS_WEIGHTS:
    if not os.path.isfile(model):
        print 'Cannot find one of the pretrained model: %s' % model 
        print 'You must either train the model first or download a readymade model.'
        sys.exit(-1)
subprocess.check_call('%s/../../generate_norb_small_missing_data.py' % SCRIPT_DIR, shell=True)

try:
    test('out_test_clean.txt', MNIST_TEST,  MODELS_PROTOS, MODELS_WEIGHTS, missing_data=False, whitened_similarity=False)
    for db in DBS:
        for mar_prob in MAR_PROBS:
            base = 'mar_%.2f' % (mar_prob)
            dataset = '%s_%s' % (db, base)
            name = 'out_test_%s.txt' % (base)
            test(name, dataset, MODELS_PROTOS, MODELS_WEIGHTS)
except KeyboardInterrupt:
    time.sleep(1)
