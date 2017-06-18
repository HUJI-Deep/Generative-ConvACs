# -*- coding: utf-8 -*-
"""

Generate MNIST missing data datasets. 
Assumes https://github.com/HUJI-Deep/Generative-ConvACs/blob/master/exp/mnist/generate_mnist_missing_data.py
has already been called.

Created on Mon May 15 19:07:47 2017

@author: ronent
"""

import numpy as np
import os
from PIL import Image
from os.path import join

eps = 0.0001


def load_image(path, image_shape, scale=255.0):
    img = np.float32(Image.open(path)) / scale
    if len(image_shape) == 3:
        c = image_shape[2]
        return img[:,:,:c]
    else:
        return img
    

def load_imgs_from_dir(imgs_dir, img_shape, index_fname, scale=255.0):
    input_dim = np.prod(img_shape)
    if not os.path.exists(imgs_dir):
        raise IOError('Error- %s doesn\'t exist!' % imgs_dir)
    raw_imgs_data = np.loadtxt(os.path.join(imgs_dir,index_fname),delimiter=' ',dtype=str)
    labels = raw_imgs_data[:,1].astype(np.uint8)
    total_images = raw_imgs_data.shape[0]
    imgs = np.zeros((total_images,input_dim))
    for idx in np.arange(total_images):
        imgs[idx,:] = load_image(os.path.join(imgs_dir,raw_imgs_data[idx][0]), img_shape, scale).reshape(input_dim)
    return imgs.astype('float32'), labels 

def convert_missing_dataset(dataset_dir):
    X,Y = load_imgs_from_dir(dataset_dir, img_shape=(28,28), index_fname='index.txt', scale=255.0)
    X = X - 0.5 # normalize
    one_hot_Y = one_hot(Y)
    X_Y = np.hstack((X,one_hot_Y))
    X_mask,_ = load_imgs_from_dir(dataset_dir, img_shape=(28,28), index_fname='index_mask.txt', scale=255.0)
    X_mask_ext = np.hstack((X_mask,np.zeros((X_mask.shape[0],10))))
    return X_Y,X_mask_ext


def one_hot(label_vec):
    K = np.max(label_vec) + 1
    m = len(label_vec)
    one_hot_labels = np.zeros((K,m))
    one_hot_labels[label_vec,np.arange(m)] = 1
    return one_hot_labels.transpose()

def mnist_spn_opt(data_dir, normalize=True, valid=True):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28))

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28))

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))
  
    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    if normalize:
        trX = (trX.astype(np.float) / 255) - 0.5
        teX = (teX.astype(np.float) / 255) - 0.5

    if valid:
        vaX = trX[50000:,:]
        vaY = trY[50000:]
        trX = trX[:50000,:]
        trY = trY[:50000]
        return trX, teX, trY, teY, vaX,vaY
    return trX, teX, trY, teY
    
    
if __name__ == "__main__":
    
    DATA_DIR = 'data'
    
    try:
        os.makedirs(DATA_DIR)
    except:
        pass
    
    mnist_test = join(DATA_DIR,'ln_mnist_test.txt')
    mnist_train = join(DATA_DIR,'ln_mnist_train.txt')
    mnist_val = join(DATA_DIR,'ln_mnist_val.txt')
    
    if (not os.path.exists(mnist_train) and 
    not os.path.exists(mnist_test) and 
    not os.path.exists(mnist_val)) :
        print "Generating MNIST for spn-opt..."
        trX, teX, trY, teY,  vaX,vaY = mnist_spn_opt(DATA_DIR, normalize=True, valid=True)
        
        
        one_hot_teY = one_hot(teY)
        one_hot_vaY = one_hot(vaY)
        one_hot_trY = one_hot(trY)
        teX = np.hstack((teX,one_hot_teY))
        vaX = np.hstack((vaX,one_hot_vaY))
        trX = np.hstack((trX,one_hot_trY))
        np.savetxt(mnist_test,teX, fmt='%.4f',delimiter=',')
        np.savetxt(mnist_train,trX, fmt='%.4f',delimiter=',')
        np.savetxt(mnist_val,vaX, fmt='%.4f',delimiter=',')

    
    print "Converting missing data MNIST to spn-opt format..."

    MNIST_TEST = 'mnist_test_image_data'
    DBS = [MNIST_TEST]
    MAR_PROBS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    RECT_NUMS = [1, 2, 3]
    RECT_WIDTHS = [7, 11, 15]
    

    
    for db in DBS:
        for mar_prob in MAR_PROBS:
            new_db = '%s_mar_%.2f' % (db, mar_prob)
            full_path = os.path.join(DATA_DIR, new_db);
            save_db_path = join(DATA_DIR,'%s_im.data' % (new_db))
            save_mask_db_path = join(DATA_DIR,'%s_mask.data' % (new_db))
            if os.path.isdir(full_path):
                if not os.path.exists(save_db_path) and not os.path.exists(save_mask_db_path):
                    print "Processing ", new_db
                    X_Y,X_mask_ext = convert_missing_dataset(full_path)
                    np.savetxt(save_db_path,X_Y, fmt='%.4f',delimiter=',')
                    np.savetxt(save_mask_db_path,X_mask_ext, fmt='%.1f',delimiter=',')
                
        for rect_num in RECT_NUMS:
            for rect_width in RECT_WIDTHS:
                new_db = '{0}_minrects_{1}_maxrects_{1}_minwidth_{2}_maxwidth_{2}'.format(db, rect_num, rect_width)
                full_path = os.path.join(DATA_DIR, new_db)
                save_db_path = join(DATA_DIR,'%s_im.data' % (new_db))
                save_mask_db_path = join(DATA_DIR,'%s_mask.data' % (new_db))
                if os.path.isdir(full_path):
                    if not os.path.exists(save_db_path) and not os.path.exists(save_mask_db_path):
                        print "Processing ", new_db
                        X_Y,X_mask_ext = convert_missing_dataset(full_path)
                        np.savetxt(save_db_path,X_Y, fmt='%.4f',delimiter=',')
                        np.savetxt(save_mask_db_path,X_mask_ext, fmt='%.1f',delimiter=',')
    
    print "Done."
    
