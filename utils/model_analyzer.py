#!/usr/bin/env python
import argparse
import os
import numpy as np
from os.path import isfile
from os.path import join
import sys, signal
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from time import time
import itertools
from multiprocessing import Pool
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
def abspath_to_resource(path):
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))
os.environ['GLOG_minloglevel'] = '2' # Hides Caffe's debug printing...
# Dynamically load the correct Caffe from our submodule
#import imp
#caffe = imp.load_source('caffe', abspath_to_resource('../deps/simnets/python/caffe/__init__.py'))
import caffe

######### External Code ##########
'''
Code for im2col modified from Standford's CS231n Course. License:
The MIT License (MIT)

Copyright (c) 2015 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1, precomputed_indices=None):
    """ An implementation of im2col based on some fancy indexing """
    x_padded = None
    if padding > 0:
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    else:
        x_padded = np.copy(x)

    if precomputed_indices is None:
        k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    else:
        k, i, j = precomputed_indices

    cols = x_padded[k, i, j]
    return cols
######### End of External Code ##########


def stable_normalize_logspace_mat(X_in):
    h,w = np.shape(X_in)
    c = np.max(X_in, axis=1)
    X_shifted = X_in - np.tile(c,(w,1)).transpose()
    exp_X_shifted = np.exp(X_shifted)
    sums = np.sum(exp_X_shifted, axis=1)
    return exp_X_shifted / (np.tile(sums,(w,1)).transpose())

def get_image_data_and_labels(index_file, get_full_path=True, as_list=True):
    if not os.path.exists(index_file):
        print 'Error, no index file at path ', index_file
        return [],[]
      
    index_file_dir = os.path.dirname(index_file)
    data = np.genfromtxt(index_file, dtype='str')
    labels = data[:,1].astype(int)
    if as_list:
        im_data= list(data[:,0])
    else:
        im_data = data[:,0]

    if get_full_path:
        im_data_f = [join(index_file_dir,im) for im in im_data ] 
        if not as_list:
            im_data_f = np.array(im_data_f)
    else:
        im_data_f = im_data
    
    return im_data_f,labels

def hash_bool_array(x):
    h = 0 # Since this method is only useful for arrays < 20 then no need to use longs
    for i in xrange(x.shape[0]):
        h = (h << 1) + x[i]
    return h

def bool_value(x):
    if x == 'y':
        return True
    elif x == 'n':
        return False
    else:
        raise ValueError("Bool parameter must be either 'y' or 'n'.")

def init_worker(num_instances, kernel_h, kernel_w, pad, stride, indices, pdfs):
    global g_num_instances
    g_num_instances = num_instances
    global g_kernel_h
    g_kernel_h = kernel_h
    global g_kernel_w
    g_kernel_w = kernel_w
    global g_pad
    g_pad = pad
    global g_stride
    g_stride = stride
    global g_indices
    g_indices = indices
    global g_pdfs
    g_pdfs = pdfs
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    np.random.seed(None)

def whiten_similarity(params):
    i, processed_data = params
    cols = im2col_indices(processed_data, g_kernel_h, g_kernel_w, g_pad, g_stride, g_indices).transpose()
    masks = ~np.isnan(cols)
    marginal_xs = [None] * cols.shape[0]
    hashs = [None] * cols.shape[0]
    sim_out = np.zeros((g_num_instances, cols.shape[0]))
    for k in xrange(cols.shape[0]):
        hashs[k] = hash_bool_array(masks[k])
        x = cols[k]
        marginal_xs[k] = x[masks[k]]
    for j in xrange(g_num_instances):
        instance_pdfs = g_pdfs[j]
        for k in xrange(cols.shape[0]):
            marginal_x = marginal_xs[k]
            if marginal_x.shape[0] == 0:
                # When marginalizing over all vars the probability is 1 (0 in logspace)
                continue
            rv = instance_pdfs[hashs[k]]
            logprob = rv.logpdf(marginal_x)
            sim_out[j, k] = logprob
    return (i, sim_out)

class ModelAnalyzer(object):
    def __init__(self, net_fn, param_fn, out_layer_name, num_classes, raw_scale=1, is_color=True, use_whitened_similarity=False):
        sys.stderr.write('Loading model...\n')
        if is_color:
            self._channel_swap = (2,1,0)
            self._base_net = caffe.Classifier(net_fn,param_fn,channel_swap = self._channel_swap,raw_scale=raw_scale)
        else:
            self._channel_swap = 0
            self._base_net = caffe.Classifier(net_fn,param_fn,raw_scale=raw_scale)
        self._out_layer_name = out_layer_name
        self._K = num_classes
        self._tested = False
        self._is_color = is_color
        self._use_whitened_similarity = use_whitened_similarity
        self._batch_size = self._base_net.blobs['data'].data.shape[0]
        if self._use_whitened_similarity:
            sys.stderr.write('Processing whitened similarity...\n')
            similiarity_index, similarity_layer = next(((i, l) for i, l in enumerate(self._base_net.layers) if l.type == 'Similarity'))
            self._after_similarity_name = self._base_net._layer_names[similiarity_index + 1]
            self._similarity_name = self._base_net._layer_names[similiarity_index]
            self._similarity_output = self._base_net.top_names[self._similarity_name][0]
            self._similarity_output_shape = self._base_net.blobs[self._similarity_output].data.shape
            self._whitening_name = self._base_net.bottom_names[self._similarity_name][0]
            whitening_index = list(self._base_net._layer_names).index(self._whitening_name)
            self._last_preprocess_layer_index = whitening_index - 1
            self._last_preprocess_layer = self._base_net._layer_names[self._last_preprocess_layer_index]
            self._whitening_input = self._base_net.bottom_names[self._whitening_name][0]
            self._whitening_input_shape = self._base_net.blobs[self._whitening_input].data.shape
            whitening_layer = self._base_net.layers[whitening_index]
            self._patch_shape = whitening_layer.blobs[0].data.shape[1:]
            W = whitening_layer.blobs[0].data
            W = W.reshape((W.shape[0], -1))
            b = whitening_layer.blobs[1].data
            b = b.reshape((b.shape[0], 1))
            # Wy + b = x => Wy = x - b => y = W^-1 (x - b) = W^-1 x - W^-1 b
            # y = Bx + c, where x ~ N(mu, sigma).
            # Then y ~ N(c + B*mu, B * sigma * B^T)
            B = np.linalg.pinv(W)
            c = - np.dot(B, b)
            mus = np.squeeze(similarity_layer.blobs[0].data).transpose()
            new_mus = c + np.dot(B, mus)
            new_mus = map(lambda x: x.flatten(), np.split(new_mus, new_mus.shape[1], axis=1))
            sigmas = (1.0 / np.squeeze(similarity_layer.blobs[1].data)).transpose()
            sigmas = np.split(sigmas, sigmas.shape[1], axis=1) # each sigma in its own array
            new_sigmas = map(lambda x: np.dot(B, np.dot(np.diag(x.flatten()), B.transpose())), sigmas)
            self._corrected_means = new_mus
            self._corrected_covs = new_sigmas
            self._num_instances = len(sigmas)
            self._pdfs = None
            with open(param_fn, 'rb') as f:
                net = caffe.proto.caffe_pb2.NetParameter()
                net.MergeFromString(f.read())
                self._conv_param = next((l for l in net.layer if l.name == self._whitening_name)).convolution_param
                if len(self._conv_param.kernel_size) > 0:
                    self._conv_param.kernel_h = self._conv_param.kernel_size[0]
                    self._conv_param.kernel_w = self._conv_param.kernel_size[0]
                if len(self._conv_param.pad) == 0:
                    self._conv_param.pad.append(0)
                if len(self._conv_param.stride) == 0:
                    self._conv_param.stride.append(1)
                temp_shape = self._whitening_input_shape[1:]
                self._im2col_indices = get_im2col_indices(temp_shape,
                    self._conv_param.kernel_h, self._conv_param.kernel_w, self._conv_param.pad[0], self._conv_param.stride[0])
    
    def classify(self,X,Y, use_normalized=True, mask=None):
        if self._use_whitened_similarity:
            self.precompute_marginals()
            self._pool = Pool(initializer=init_worker, initargs=(self._num_instances,
                self._conv_param.kernel_h, self._conv_param.kernel_w, self._conv_param.pad[0],
                self._conv_param.stride[0], self._im2col_indices, self._pdfs))
        probs, preds = self.collect_probs(X, Y, use_normalized, mask=mask)
        self._prob_mat = probs
        self._Y_hat = preds
        self._Y = Y
        self._tested = True
        if self._use_whitened_similarity:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._pdfs = None
    def precompute_marginals(self):
        sys.stderr.write('Precomputing marginals...\n')
        self._pdfs = [None] * self._num_instances
        # precomputing all possible marginals
        for i in xrange(self._num_instances):
            mean = self._corrected_means[i]
            cov = self._corrected_covs[i]
            self._pdfs[i] = [None] * (2 ** mean.shape[0])
            for marginal_pattern in itertools.product([False, True], repeat=mean.shape[0]):
                marginal_length = marginal_pattern.count(True)
                if marginal_length == 0:
                    continue
                m = np.array(marginal_pattern)
                marginal_mean = mean[m]
                mm = m[:, np.newaxis]
                marginal_cov = cov[np.dot(mm, mm.transpose())].reshape((marginal_length, marginal_length))
                self._pdfs[i][hash_bool_array(m)] = multivariate_normal(mean=marginal_mean, cov=marginal_cov)
    def batch_get_probs(self, use_normalized=True):
        out = None
        start_time = time()
        if not self._use_whitened_similarity:
            out = self._base_net.forward()
        else:
            self._base_net._forward(0, self._last_preprocess_layer_index)
            processed_data = self._base_net.blobs[self._whitening_input].data
            sim_out = np.zeros(self._similarity_output_shape).reshape((self._batch_size, self._num_instances, -1))
            results = self._pool.imap_unordered(whiten_similarity,
                [(i, processed_data[i]) for i in xrange(processed_data.shape[0])], chunksize=2)
            for i, (img_index, res) in enumerate(results):
                sys.stderr.write('evaluating image: %d  \r' % (i + 1))
                sys.stdout.flush()
                sim_out[img_index] = res
            sys.stderr.write('\n')
            sim_out = sim_out.reshape(self._similarity_output_shape)
            self._base_net.blobs[self._similarity_output].data[...] = sim_out
            out = self._base_net.forward(start=self._after_similarity_name)
        end_time = time()
        sys.stderr.write('Total time for last batch (in sec): %f\n' % (end_time - start_time))
        sys.stderr.write('Time per image (in sec): %f\n' % ((end_time - start_time) / self._batch_size))
        sys.stderr.write('\n')
        acts = self._base_net.blobs[self._out_layer_name].data 
        acts = acts[:,:,0,0]
        if use_normalized:
            return acts
        else:
            return stable_normalize_logspace_mat(acts)
    
    """
    Collect channel assignment probabilities of dataset samples.    
    ** Assuming batch size divides dataset size **
    """
    def collect_probs(self, X, Y, use_normalized=True, mask=None):
        batch_size = self._base_net.blobs['data'].data.shape[0]
        dataset_size = len(Y)
        num_batches = dataset_size/batch_size
        all_probs = np.zeros((dataset_size, self._K))
        num_batches = dataset_size / batch_size
        for j in range(num_batches):
            start = j*batch_size
            sys.stderr.write('Testing on batch %d of %d\n' % (j + 1, num_batches))
            batch_data = map(lambda x: self._base_net.transformer.preprocess('data',caffe.io.load_image(x, color=self._is_color)), X[start:(start+batch_size)])
            batch_data = np.stack(batch_data)
            if mask:
                batch_masks = np.stack(map(lambda x: caffe.io.load_image(x, color=self._is_color).transpose((2,0,1)), mask[start:(start+batch_size)]))
                if self._is_color:
                    batch_masks = batch_masks[:, self._channel_swap, :, :]
                batch_data[batch_masks == 1] = np.nan # Missing data will turn into NaN and the rest will stay the same
            self._base_net.blobs['data'].data[...] = batch_data
            probs = self.batch_get_probs(use_normalized)
            all_probs[start:(start+batch_size),:] = probs
        labels = np.argmax(all_probs, axis=1).astype(int)
        return (all_probs, labels)
    
    def get_classification_hist(self):
        if not self._tested:
            print 'You must run classifer first on data to obtain statistics'
            return []
        wrong_preds = self._Y[self._Y != self._Y_hat]
        wr_hist,bins = np.histogram(wrong_preds,np.arange(self._K+1))
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1,1,1)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        n_wr_hist = wr_hist.astype(float) / np.sum(wr_hist)
        ax.bar(center, n_wr_hist, align='center', width=width)
        return n_wr_hist

if __name__ == '__main__':
    
    description = ('Script for analyzing performance of a trained Caffe network')
        
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "prototxt_path",
        help="path to caffe prototxt file. This should be a deploy prototxt file"
        " (see Caffe documentation)",
        type=str
    )
   
    parser.add_argument(
        "caffemodel_path",
        help="path to caffemodel",
        type=str
    )   
    parser.add_argument(
        "data_index_file",
        help="path to index file of data",
        type=str
    )
    parser.add_argument(
        "--missing_data_index_file",
        help="path to index file of missing data masks",
        type=str
    )
    parser.add_argument(
        "out_layer",
        help="name of output layer to be used for classification.",
        type=str
    )
    parser.add_argument(
        "--starts_with_whitened_similarity",
        help="If true then assumes the model is probabilistic and assumes it begins with conv + similarity. True is represented by 'y', and false otherwise.",
        type=bool_value, default=False
    )
    args = parser.parse_args()
    
    script_dir = os.getcwd()
    if not isfile(join(script_dir,args.caffemodel_path)) or not isfile(join(script_dir,args.prototxt_path)):
        print 'Caffemodel\prototxt dont exist at specified path. Exiting...'
        exit(-1)
    if not isfile(join(script_dir,args.data_index_file)):
        print 'Data index file doesnt exist at specified path. Exiting...'
        exit(-1)    
    if args.missing_data_index_file and len(args.missing_data_index_file) > 0 and not isfile(join(script_dir,args.missing_data_index_file)):
        print 'Missing data index file doesnt exist at specified path. Exiting...'
        exit(-1) 
    
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    # Example use
    
    # Load validation and test data 
    X, Y = get_image_data_and_labels(args.data_index_file)
    M = None
    if args.missing_data_index_file and len(args.missing_data_index_file) > 0:
        M, _ = get_image_data_and_labels(args.missing_data_index_file)

    # Load pre-trained network
    net_fn = args.prototxt_path
    param_fn = args.caffemodel_path
    net_w = ModelAnalyzer(net_fn, param_fn, args.out_layer, np.max(Y)+1, use_whitened_similarity=args.starts_with_whitened_similarity)
    net_w.classify(X, Y, mask=M)
    
    acc = float(np.sum(net_w._Y_hat == net_w._Y)) / len(net_w._Y)
    print 'Accuracy: %f ' % (acc)


