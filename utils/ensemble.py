#!/usr/bin/env python
import numpy as np
import argparse, os, sys
import model_analyzer as ma
caffe = ma.caffe

class Ensemble(object):
     def __init__(self, net_fns, param_fn_list, out_layer_name, num_classes, raw_scale=1, is_color=True, use_whitened_similarity=False):
         self._models = []
         if not isinstance(net_fns, (list, tuple)):
            net_fn = net_fns
            net_fns = [net_fn for i in range(len(param_fn_list))]
         if len(net_fns) == 1 and len(param_fn_list) > 1:
            net_fn = net_fns[0]
            net_fns = [net_fn for i in range(len(param_fn_list))]
         if len(net_fns) != len(param_fn_list):
            raise ValueError
         for net_fn, param_fn in zip(net_fns, param_fn_list):
             net = ma.ModelAnalyzer(net_fn,param_fn,out_layer_name,num_classes,
                            raw_scale, is_color, use_whitened_similarity)                
             self._models.append(net)
         self._size = len(self._models)
     
     def classify(self, X,Y, score_type='avg', mask=None):
         if score_type == 'avg':
             return self.classify_avg(X,Y, mask)
         elif score_type == 'vote':
             return self.classify_vote(X,Y, mask)
         else:
             raise IOError('Unknown scoring method, must be in [\'avg\',\'vote\']')
             
     def classify_avg(self, X,Y, mask=None):
         total_probs = np.zeros((len(Y),self._models[0]._K))
         for i, model in enumerate(self._models):
             sys.stderr.write('========================\n')
             sys.stderr.write('= Testing Model %d of %d =\n' % (i+1, len(self._models)))
             sys.stderr.write('========================\n\n')
             model.classify(X,Y,mask=mask)
             total_probs += model._prob_mat
         total_probs /= self._size
         labels = np.argmax(total_probs, axis=1).astype(int)
         accuracy = float(np.sum(labels == Y)) / len(Y)
         return accuracy
     
     def classify_vote(self, X,Y, mask=None):
         total_votes = np.zeros((len(Y),self._models[0]._K))
         for model in self._models:
             model.classify(X,Y,mask=mask)
             labels = np.argmax(model._prob_mat, axis=1).astype(int)
             total_votes[np.arange(len(Y)),labels] += 1
         total_pred = labels = np.argmax(total_votes, axis=1).astype(int)
         accuracy = float(np.sum(total_pred == Y)) / len(Y)
         return accuracy
         
     def get_single_model_accs(self):
         accs = []
         for model in self._models:
             if model._tested:
                 acc = float(np.sum(model._Y == model._Y_hat)) / len(model._Y)
                 accs.append(acc)
             else:
                 print "You must run classify before getting individual model accuracies" 
                 return []
         return accs


def list_of_filenames_type(s):
    out = []
    for r in s.split(','):
        out.append(r)
    return out

if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()
    description = ('Script for running ensemble of caffemodel files for a '
    'classification task')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "prototxt_path",
        help="path to caffe prototxt file. This should be a deploy prototxt file"
        " (see Caffe documentation). Could be a single file for all weights, or one"
        " prototxt file for every weight file.",
        type=list_of_filenames_type
    )
    parser.add_argument(
        "caffemodel_path_list",
        help="paths to caffemodel",
        type=list_of_filenames_type
    )   
    parser.add_argument(
        "data_index_file",
        help="path to index file of data",
        type=str
        
    )
    parser.add_argument(
        "out_layer",
        help="name of output layer to be used for classification.",
        type=str
    )
    parser.add_argument(
        "-raw_scale",
        help="Set the scale of raw features s.t. the input blob = input * scale."
        "While Python represents images in [0, 1], certain Caffe models"
        "like CaffeNet and AlexNet represent images in [0, 255] so the raw_scale"
        "of these models must be 255.", type=int, default = 1
    )
    parser.add_argument(
        "-is_color",
        help="y for color dataset, n for grayscale (default)",
        type=ma.bool_value, default=False
    )
    parser.add_argument(
        "-score_type",
        help='avg for averaging probabilities over all models and returning '
        'argmax (default), vote for voting between models.',
        type=str, default='avg'
    )
    parser.add_argument(
        "-single_accs",
        help='y to print each model\'s accuracy in addition to full ensemble '
        'accuracy. Default is n',
        type=ma.bool_value, default=False
    )
    parser.add_argument(
        "-missing_data_index_file",
        help="path to index file of missing data masks",
        type=str
    )
    parser.add_argument(
        "-starts_with_whitened_similarity",
        help="If true then assumes the model is probabilistic and assumes it begins with conv + similarity. True is represented by 'y', and false otherwise.",
        type=ma.bool_value, default=False
    )
    try:
        args = parser.parse_args()
        data_index_file = args.data_index_file
        net_fns = args.prototxt_path
        param_list = args.caffemodel_path_list
        out_layer = args.out_layer
        script_dir = os.getcwd()
        
        if not os.path.isfile(os.path.join(script_dir,args.data_index_file)):
            print 'Data index file %s does not exist at specified path. Exiting...' % os.path.join(script_dir,args.data_index_file)
            exit(-1) 
        if args.missing_data_index_file and len(args.missing_data_index_file) > 0 and not os.path.isfile(os.path.join(script_dir,args.missing_data_index_file)):
            print 'Missing data index file does not exist at specified path. Exiting...'
            exit(-1) 
        if len(net_fns) > 1 and len(net_fns) != len(param_list):
            print 'There is no matching prototxt for every weight file'
        for weight_file in param_list:
            if not os.path.isfile(os.path.join(script_dir,weight_file)):
                print 'Weights file %s does not exist at specified path. Exiting...' % os.path.join(script_dir,weight_file)
                exit(-1) 
        for net_fn in net_fns:
            if not os.path.isfile(os.path.join(script_dir,net_fn)):
                print 'Net deploy file %s does not exist at specified path. Exiting...' % os.path.join(script_dir,net_fn)
                exit(-1) 
        
        # Load data and labels
        X,Y = ma.get_image_data_and_labels(args.data_index_file)
        M = None
        if args.missing_data_index_file and len(args.missing_data_index_file) > 0:
            M, _ = ma.get_image_data_and_labels(args.missing_data_index_file)
        total_images = len(Y)
        
        net_ensemble = Ensemble(net_fns, param_list, out_layer, np.max(Y)+1, 
                                raw_scale=args.raw_scale, is_color=args.is_color,
                                use_whitened_similarity=args.starts_with_whitened_similarity)
        acc = net_ensemble.classify(X,Y,score_type=args.score_type, mask=M)
        print "Ensemble accuracy is %f" % (acc)
        if args.single_accs:
            print "Single model accuracies are: ", net_ensemble.get_single_model_accs()
    except KeyboardInterrupt:
        print 'User interrupted script!'