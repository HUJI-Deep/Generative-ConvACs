#!/usr/bin/env python
import os, subprocess


TRAIN_ITERS = 20

train_dir = os.path.dirname(os.path.realpath(__file__))
enclosed_dir = os.path.normpath(os.path.join(train_dir, '../../'))
data_dir = os.path.normpath(os.path.join(enclosed_dir, 'data')) 
spn_dir = os.path.normpath(os.path.join(train_dir, '../'))
spn_opt_dir = os.path.join(enclosed_dir,'..', '..', 'deps', 'spn-opt-discrim')
java_spn_path = os.path.join(enclosed_dir,'..', '..', 'deps', 'spn-poon-2011', 'results', 'mnist', 'models', 'mnist.mdl')

mnist_train_data = os.path.join(data_dir,'ln_mnist_train.txt')
mnist_test_data = os.path.join(data_dir,'ln_mnist_test.txt')
mnist_val_data = os.path.join(data_dir,'ln_mnist_val.txt')
model_path = os.path.join(spn_dir, 'mnist.spn.txt')
output_model_path = os.path.join(train_dir, 'spn-opt-mnist.mdl')

spn_train_script = os.path.normpath(os.path.join(spn_opt_dir, 'bin', 'online_learning'))
spn_train_cmd = [spn_train_script, '--train', mnist_train_data,
                 '--test', mnist_test_data,
                 '--valid', mnist_val_data,
                 '--model', model_path,
                 '--output_model', output_model_path,
                 '--algo "em" --lap_lambda 0.001 --stop_thred 1e-3 --num_iters %d' % (TRAIN_ITERS)]

# Generate datasets in the correct format
print "Checking whether MNIST dataset already exists..."
subprocess.check_call('%s/generate_mnist.py' % enclosed_dir, shell=True)
print "Checking whether MNIST missing data datasets already exist..."
subprocess.check_call('%s/generate_mnist_missing_data.py' % enclosed_dir, shell=True)
subprocess.check_call('python %s/convert_mnist_for_spn_opt.py' % enclosed_dir, shell=True, cwd=enclosed_dir)

if not os.path.exists(model_path):
    print "Converting model file to spn-opt format..."
    subprocess.check_call('python %s/convert_model.py %s %s' % (spn_dir,java_spn_path,model_path), 
                      shell=True, cwd=spn_dir)
                  

try:
        print '=============== TRAINING ==============='
        subprocess.check_call(' '.join(spn_train_cmd), shell=True)
        print '=============== DONE ==============='
except:
    print 'Training terminated.'
