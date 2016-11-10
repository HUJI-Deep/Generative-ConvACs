#!/usr/bin/env python


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from os.path import join
import sys
import argparse
import PIL.Image
import progressbar

eps = 0.0001


def load_image(path, image_shape, scale=255.0):
    img = np.float32(PIL.Image.open(path)) / scale
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
        sys.stdout.flush()
    return imgs.astype('float32'), labels 
    
"""
Evaluates performance of a KNN classifier on corrupted data.
Input:
trX,trY - Training data and labels to be used for KNN classification.
missing_data_dir - Directory containing corrupted data. Assuming existence of image data
indexed at index.txt and masks corresponding to each image indexed at 
index_mask.txt. Masks are binary, 1 corresponding to missing locations and 0
to observed locations.
k - number of neighbors to use in KNN classification.

Output:
prob_Y_hat - m x Y normalized votes matrix, each row being the votes for a 
test sample (over Y classes).
"""    
def knn_masked_data(trX,trY,missing_data_dir, input_shape, k):
    
    raw_im_data = np.loadtxt(join(script_dir,missing_data_dir,'index.txt'),delimiter=' ',dtype=str)
    raw_mask_data = np.loadtxt(join(script_dir,missing_data_dir,'index_mask.txt'),delimiter=' ',dtype=str)
    # Using 'brute' method since we only want to do one query per classifier
    # so this will be quicker as it avoids overhead of creating a search tree
    knn_m = KNeighborsClassifier(algorithm='brute',n_neighbors=k)
    prob_Y_hat = np.zeros((raw_im_data.shape[0],int(np.max(trY)+1)))
    total_images = raw_im_data.shape[0]
    pbar = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('\rProcessed %(value)d of %(max)d Images '), progressbar.Bar()], maxval=total_images, term_width=50).start()
    for i in range(total_images):
        mask_im=load_image(join(script_dir,missing_data_dir,raw_mask_data[i][0]), input_shape,1).reshape(np.prod(input_shape))
        mask = np.logical_not(mask_im > eps) # since mask is 1 at missing locations
        v_im=load_image(join(script_dir,missing_data_dir,raw_im_data[i][0]), input_shape, 255).reshape(np.prod(input_shape))
        rep_mask = np.tile(mask,(trX.shape[0],1))
        # Corrupt whole training set according to the current mask
        corr_trX = np.multiply(trX, rep_mask)        
        knn_m.fit(corr_trX, trY)
        prob_Y_hat[i,:] = knn_m.predict_proba(v_im)
        pbar.update(i)
    pbar.finish()
    return prob_Y_hat
    

def load_norb_small(data_dir):
    tr_path = os.path.join(data_dir,'norb_small_2D_whole_train_image_data')
    te_path = os.path.join(data_dir,'norb_small_2D_test_image_data')
    print '\nLoading train set images...'
    trX, trY = load_imgs_from_dir(tr_path, (32,32,2), 'index.txt')
    print '\nLoading test set images...'
    teX, teY = load_imgs_from_dir(te_path, (32,32,2), 'index.txt')
    
    return trX, teX, trY, teY
    
    
def load_mnist(data_dir):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,np.prod(input_shape))).astype(float) / 255

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,np.prod(input_shape))).astype(float) / 255

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))
    
    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY
    
def load_data(dataset, data_dir, input_shape):
    if dataset == "mnist":
        trX, teX, trY, teY = load_mnist(data_dir)
    else: # norb
        trX, teX, trY, teY = load_norb_small(args.data_dir)
    return trX, teX, trY, teY

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        help="Directory where the ubyte data files are contained",
        default='../data',
        type=str
    )

    parser.add_argument(
        "--missing_data_dir",
        help="Directory of missing data dataset",
        type=str
    )

    parser.add_argument(
        "--k",
        help="k to use in the k-NN classifier.",
        default=5,
        type=int
    )
    
    parser.add_argument(
        "--dataset",
        help="'mnist' or 'norb'",
        default="mnist",
        type=str
    )

    script_dir = os.path.dirname(os.path.realpath(__file__))
    args = parser.parse_args()
    
    k = args.k    
    if args.dataset == "mnist":
        Y = 10  
        input_shape = (28,28)
    elif args.dataset == "norb":
        Y = 5
        input_shape = (32,32,2)
    else:
        print "Error: dataset argument must be either 'mnist' or 'norb', got ", args.dataset
        sys.exit(-1)
    trX, teX, trY, teY = load_data(args.dataset, args.data_dir, input_shape)
    
    # Load missing data labels
    missing_data_dir = args.missing_data_dir
    Y_true = np.loadtxt(join(script_dir,missing_data_dir,'index.txt'),delimiter=' ',dtype=str)[:,1].astype(np.uint8)
    m= Y_true.shape[0]        

    # Get KNN classification results over missing data dataset
    probs = np.zeros((m,Y))
    f_path = join(script_dir,missing_data_dir)
    probs = knn_masked_data(trX, trY, f_path,input_shape, k)
    preds = np.argmax(probs, axis=1)
    acc = float(np.sum((preds == Y_true))) / m
    print '%s %d %.4f' % (os.path.basename(missing_data_dir),k, acc)
    
    
    
    