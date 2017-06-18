#!/usr/bin/env python
import os, subprocess, time


test_dir = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.normpath(os.path.join(test_dir, '../train'))
enclosed_dir = os.path.normpath(os.path.join(train_dir, '../../'))
data_dir = os.path.normpath(os.path.join(enclosed_dir, 'data')) 
spn_dir = os.path.normpath(os.path.join(train_dir, '../'))
spn_opt_dir = os.path.join(enclosed_dir,'..', '..', 'deps', 'spn-opt-discrim')
TEST_SCRIPT = os.path.normpath(os.path.join(spn_opt_dir, 'bin', 'test_model'))


MNIST_TEST = 'mnist_test_image_data'
DBS = [MNIST_TEST]
MAR_PROBS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
RECT_NUMS = [1, 2, 3]
RECT_WIDTHS = [7, 11, 15]
MODEL_PATH = os.path.join(train_dir, 'spn-opt-mnist.mdl')

try:
    for db in DBS:
        for mar_prob in MAR_PROBS:
            new_db = '%s_mar_%.2f' % (db, mar_prob)
            full_path = os.path.join(data_dir, new_db)
            output_file = os.path.join(test_dir,'test_%s.out' % (new_db))
            if not os.path.exists(output_file):
                test_cmd = [TEST_SCRIPT,
                '--test', full_path + '_im.data',
                '--masks', full_path + '_mask.data',
                '--model', MODEL_PATH,
                '--algo', '"em"',
                '--lap_smooth 0.001 --stop_thred 1e-3',
                '--num_iters 1',
                '>', output_file]
                cmd = ' '.join(test_cmd)
                print cmd
                subprocess.check_call(cmd, shell=True, cwd=test_dir)
            else:
                print "%s exits, continuing..." % (output_file)
        for rect_num in RECT_NUMS:
            for rect_width in RECT_WIDTHS:
                new_db = '{0}_minrects_{1}_maxrects_{1}_minwidth_{2}_maxwidth_{2}'.format(db, rect_num, rect_width)
                full_path = os.path.join(data_dir, new_db)
                output_file = os.path.join(test_dir,'test_%s.out' % (new_db))
                if not os.path.exists(output_file):
                    test_cmd = [TEST_SCRIPT,
                '--test', full_path + '_im.data',
                '--masks', full_path + '_mask.data',
                '--model', MODEL_PATH,
                '--algo', '"em"',
                '--lap_smooth 0.001 --stop_thred 1e-3',
                '--num_iters 1',
                '>', output_file]
                    cmd = ' '.join(test_cmd)
                    print cmd
                    subprocess.check_call(cmd, shell=True, cwd=test_dir)
                else:
                    print "%s exits, continuing..." % (output_file)
except KeyboardInterrupt:
    time.sleep(1)
    print 'Error while generating missing data datasets!'
