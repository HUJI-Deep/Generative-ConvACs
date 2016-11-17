#!/usr/bin/env python
import os, shutil, time, sys, requests, argparse, subprocess, signal
from multiprocessing import Pool, cpu_count
import numpy as np
import scipy, scipy.misc
from scipy import ndimage

BASE_NORB_URL = 'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/'
SM_NORB_BASE_TR = BASE_NORB_URL + 'smallnorb-5x46789x9x18x6x2x96x96'
SM_NORB_BASE_TE = BASE_NORB_URL + 'smallnorb-5x01235x9x18x6x2x96x96'
SM_NORB_TR_DATA = SM_NORB_BASE_TR + '-training-dat.mat.gz'
SM_NORB_TR_LABELS = SM_NORB_BASE_TR + '-training-cat.mat.gz'
SM_NORB_TR_INFO = SM_NORB_BASE_TR + '-training-info.mat.gz'
SM_NORB_TE_DATA = SM_NORB_BASE_TE + '-testing-dat.mat.gz'
SM_NORB_TE_LABELS = SM_NORB_BASE_TE + '-testing-cat.mat.gz'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

IM_FORMAT = '.png'
MAT_SHAPE = (24300,2,96,96)
NUM_CLASSES = 5
M_VAL = 4000
DOWNSCALE_FACTOR = (1. / 3)
TR_CATS = [4,6,7,8,9] 


def crop_and_rescale(im, cr_dim):
    h = im.shape[0]
    w = im.shape[1]
    rescale_factor = float(h) / ( h - (2*cr_dim) )# assuming square input
    if len(im.shape) == 3:
        cr_im = im[cr_dim:(h-cr_dim),cr_dim:(w-cr_dim),:]
        rescaled_im = np.zeros((h,w,3))
        downscaled_im = np.zeros((int(h*DOWNSCALE_FACTOR),int(w*DOWNSCALE_FACTOR),3))
        for c in range(im.shape[2]):
            rescaled_im[:,:,c] = ndimage.interpolation.zoom(cr_im[:,:,c], rescale_factor)
            downscaled_im[:,:,c] = ndimage.interpolation.zoom(rescaled_im[:,:,c], DOWNSCALE_FACTOR)
    else:
        cr_im = im[cr_dim:(h-cr_dim),cr_dim:(w-cr_dim)]
        rescaled_im = ndimage.interpolation.zoom(cr_im, rescale_factor)
        downscaled_im = ndimage.interpolation.zoom(rescaled_im, DOWNSCALE_FACTOR)
    return downscaled_im
    
def rotate_im(im, degree):
    h = im.shape[0]
    w = im.shape[1]
    # to get the background color simply by taking mean of first row
    if len(im.shape) == 3:
        av_bg_val = np.mean(im[0,0,:])
    else:
        av_bg_val = np.mean(im[0,:])
    rot_im = ndimage.interpolation.rotate(im, degree, reshape=False, mode='constant', cval=av_bg_val)
    if len(im.shape) == 3:
        downscaled_im = np.zeros((int(h*DOWNSCALE_FACTOR),int(w*DOWNSCALE_FACTOR),3))
        for c in range(im.shape[2]):
            downscaled_im[:,:,c] = ndimage.interpolation.zoom(rot_im[:,:,c], DOWNSCALE_FACTOR)
    else:
        downscaled_im = ndimage.interpolation.zoom(rot_im, DOWNSCALE_FACTOR)
    return downscaled_im

def get_val_indices_uniform(m_total, m_val):
    all_idxs = np.arange(m_total)    
    samps_per_class = m_val / NUM_CLASSES
    val_idxs = np.array([])
    for i in range(NUM_CLASSES):
        all_class_idxs = all_idxs[( all_idxs % NUM_CLASSES == i)]
        sel_class_idxs = np.random.choice(all_class_idxs, samps_per_class, replace=False)
        val_idxs = np.concatenate((val_idxs,sel_class_idxs))
    np.random.shuffle(val_idxs)
    return val_idxs.astype(np.uint)

def get_val_indices(m_total, m_val, info_mat):
    all_idxs = np.arange(m_total)    
    val_idxs = np.array([])
    for i in range(NUM_CLASSES):
        cat_for_val = np.random.choice(TR_CATS,1)[0]
        all_class_idxs = all_idxs[( all_idxs % NUM_CLASSES == i)]
        class_info = info_mat[all_class_idxs]
        sel_class_idxs = np.where(class_info[:,0] == cat_for_val)[0]
        val_idxs = np.concatenate((val_idxs,all_class_idxs[sel_class_idxs]))
    np.random.shuffle(val_idxs)
    return val_idxs.astype(np.uint)
    
      
def download_data(data_url, directory='.'):
    file_name = '%s/%s' % (directory, data_url.split('/')[-1])
    if os.path.isfile(file_name):
        return file_name
    try:
        temp_file_name = file_name + '~'
        with open(temp_file_name, "wb") as f:
            print '#######'
            print "Downloading %s" % os.path.basename(file_name)
            start = time.clock()
            response = requests.get(data_url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None: # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for chunk in response.iter_content(512*1024):
                    dl += len(chunk)
                    f.write(chunk)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s] %s KBps" % ('=' * done, ' ' * (50-done), dl//(8*1024*(time.clock() - start))))
                    sys.stdout.flush()
            print '\nDone'
        os.rename(temp_file_name, file_name)
    except:
        if os.path.isfile(temp_file_name):
            os.remove(temp_file_name)
        raise
    
    return file_name

def unarchive_data(file_name, directory='.'):
    # unarchive data and return the list of files
    print 'Unarchiving: %s' % os.path.basename(file_name)
    new_file = '%s/%s' % (directory, os.path.basename(os.path.splitext(file_name)[0]))
    if os.path.isfile(new_file):
        return [new_file]
    contents = [new_file]
    subprocess.check_call(['gzip', '-d', '-k', file_name])
    return contents

class DBWrapper(object):
    @classmethod
    def supported_db_types(cls):
        return [ 'image_data'] # supporting only image_data and not leveldb or lmdb at this stage

    def __init__(self, db_type='image_data'):
        self._db_type = db_type
        self._db = None
        self._txn = None
        if db_type not in DBWrapper.supported_db_types():
            raise Exception('Unsupported db type: ' + db_type)
    
    def open(self, db_name, mode):
        if self._db is not None:
            raise Exception('DB already open!')
        if mode not in ['r', 'w']:
            raise Exception('Unsupported mode: ' + mode)
        
        if self._db_type == 'image_data':
            self._db_name = db_name
            self._db_mode = mode
            self._db_index = os.path.join(db_name, 'index.txt')
            if mode == 'r':
                if not os.path.isdir(self._db_name) or not os.path.isfile(self._db_index):
                    raise Exception("DB doesn't exist!")
                with open(self._db_index, 'r') as f:
                    self._index_data = f.read()
            elif mode == 'w':
                try:
                    os.mkdir(self._db_name)
                except:
                    pass
                if os.path.isfile(self._db_index):
                    with open(self._db_index, 'r') as f:
                        self._index_data = f.read()
                else:
                    self._index_data = ''
    
    def close(self):
        if self._db is not None:
            self._db.close()
    def commit(self):
        if self._txn is not None:
            self._txn.commit()
    def put(self, key, value = 0): 
        if self._db_type == 'image_data':
            
            with open(self._db_index, 'a') as f:
                if len(self._index_data) > 0:
                    self._index_data += '\n'
                    f.write('\n')
                record = '%s %d' % (key, value) 
                f.write(record)
                self._index_data += record

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    np.random.seed(None)

def process_file(params):
    index, data, base_filename, db_name, C, aug_data = params
    label = index % NUM_CLASSES
    if C==1:
        orig_im = data[0,:,:]
        im = ndimage.interpolation.zoom(orig_im, DOWNSCALE_FACTOR)
    elif C==2:
        im = np.zeros((int(MAT_SHAPE[2]*DOWNSCALE_FACTOR),int(MAT_SHAPE[3]*DOWNSCALE_FACTOR),3))
        orig_im = np.zeros((MAT_SHAPE[2],MAT_SHAPE[3],3))
        im[:,:,0] =  ndimage.interpolation.zoom(data[0,:,:], DOWNSCALE_FACTOR)
        im[:,:,1] =  ndimage.interpolation.zoom(data[1,:,:], DOWNSCALE_FACTOR)
        orig_im[:,:,0] =  data[0,:,:]
        orig_im[:,:,1] =  data[1,:,:]
    else:
        print "Error in reading data to db- number of channels must be 1 or 2"
    im_name = '%s_%d%s' % (base_filename, index,IM_FORMAT)
    scipy.misc.toimage(im, cmin=0.0, cmax=255.0).save(os.path.join(db_name,im_name))
    im_names = [im_name]
    if aug_data:
        degrees = [-20, -10, 10, 20]
        crop_dims = [2, 4, 6, 8]
        for i, degree in enumerate(degrees):
            im_name = '%s_%d_%d%s' % (base_filename,index,degree,IM_FORMAT)
            im_names.append(im_name)
            rot_im = rotate_im(orig_im, degree)
            scipy.misc.toimage(rot_im, cmin=0.0, cmax=255.0).save(os.path.join(db_name,im_name))
        for i, crop_dim in enumerate(crop_dims):
            im_name = '%s_%d_%d%s' % (base_filename,index,crop_dim,IM_FORMAT)
            im_names.append(im_name)
            cr_im = crop_and_rescale(orig_im, crop_dim)        
            scipy.misc.toimage(cr_im, cmin=0.0, cmax=255.0).save(os.path.join(db_name,im_name))
    return label, im_names

def read_data_to_db(mat_file, mat_shape, indices, db, C=1, aug_data=False):
    pool = Pool(cpu_count() * 2, init_worker)
    base_filename = os.path.basename(db._db_name)
    with open(mat_file[0], 'rb') as f:
        b = f.read()
        data = np.fromstring(b[24:], dtype=np.uint8).reshape(mat_shape)
    results = pool.imap_unordered(process_file,
        [(index, data[index], base_filename, db._db_name, C, aug_data) for index in indices],
        chunksize=10)
    for j, (label, im_names) in enumerate(results):
        sys.stdout.write('\r Generating %d of %d Images' % (j + 1, len(indices)))
        sys.stdout.flush()
        for im_name in im_names:
            db.put(im_name,label)
        
def create_db( db_path, mat_file, mat_shape, indices, C=1, aug_data=False ): 
    if not filter(lambda x: not os.path.isdir(x), [db_path]): 
        print '%s already exists' % (db_path)
        return

    try:
        if not os.path.isdir(db_path):
            print 'Creating db at ',db_path
            db = DBWrapper(db_type = 'image_data')
            db.open(db_path, 'w') 
            read_data_to_db(mat_file, mat_shape, indices, db, C, aug_data)
            
            
        else:
            print '%s already exists' % (db_path)
            
    except:
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        raise
    finally: 
        db.commit()
        db.close()

def num_channels(c):
    c = int(c)
    if c not in [1,2]:
        raise ValueError("number of channels must be either 1 or 2")
    return c

if __name__ == '__main__':
    description = ('Script for generating small NORB dataset, with the'
    'options of augmenting the data with scaling and rotation, and the option'
    ' of taking either monocular or binocular image.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--num_channels",
        help="1 to take only left monocular image, 2 for binocular image."
        " Will be saved in .png format where channel 0,1 are the 2 channels"
        " and third channel is empty.",
        type=num_channels,default=2
    )
    parser.add_argument(
        "--aug_data",
        help="'y' to augment data with rotations in {-20,-10,10,20} degrees "
        " and scaling (up to 15%%)  for each image.",
        type=str,default="n"
    )
    
    args = parser.parse_args()
    C = args.num_channels
    aug_data = (args.aug_data == "y")
    aug_str = "aug_" if aug_data else ""
    db_type = 'image_data'

    # Paths of datasets
    whole_train_db = os.path.join(SCRIPT_DIR, 'data' , ('%snorb_small_%dD_whole_train_' % (aug_str,C)) + db_type)
    train_db = os.path.join(SCRIPT_DIR, 'data' , ('%snorb_small_%dD_train_' % (aug_str,C)) + db_type)
    validation_db = os.path.join(SCRIPT_DIR, 'data' ,('%snorb_small_%dD_validation_' % (aug_str,C)) + db_type)
    test_db = os.path.join(SCRIPT_DIR, 'data' , ('norb_small_%dD_test_' % (C))+ db_type)
    if not False in [os.path.isdir(db) for db in [whole_train_db, train_db, validation_db, test_db]]:
        print 'All required datasets are present.'
        sys.exit(0)

    print 'Generating databases'
    # Make sure the data directory exists
    try:
        os.mkdir(os.path.join(SCRIPT_DIR, 'data'))
    except:
        pass
    # Download data
    train_img_gz_file = download_data(SM_NORB_TR_DATA, os.path.join(SCRIPT_DIR, 'data'))
    train_label_gz_file = download_data(SM_NORB_TR_LABELS, os.path.join(SCRIPT_DIR, 'data'))
    train_info_gz_file = download_data(SM_NORB_TR_INFO, os.path.join(SCRIPT_DIR, 'data'))
    test_img_gz_file = download_data(SM_NORB_TE_DATA, os.path.join(SCRIPT_DIR, 'data'))
    test_label_gz_file = download_data(SM_NORB_TE_LABELS, os.path.join(SCRIPT_DIR, 'data'))
    
    train_image_mat_file = unarchive_data(train_img_gz_file, directory=os.path.join(SCRIPT_DIR, 'data'))
    test_image_mat_file = unarchive_data(test_img_gz_file, directory=os.path.join(SCRIPT_DIR, 'data'))
    
    train_label_mat_file = unarchive_data(train_label_gz_file, directory=os.path.join(SCRIPT_DIR, 'data'))
    test_label_mat_file = unarchive_data(test_label_gz_file, directory=os.path.join(SCRIPT_DIR, 'data'))
    
    train_info_mat_file = unarchive_data(train_info_gz_file, directory=os.path.join(SCRIPT_DIR, 'data'))[0]
    
    with open(train_info_mat_file, 'rb') as f:
        b = f.read()
        tr_info = np.fromstring(b[20:], dtype=np.uint32).reshape(24300,4)
    
    val_idxs = get_val_indices(MAT_SHAPE[0], M_VAL, tr_info)
    train_idxs =  np.setdiff1d(range(MAT_SHAPE[0]), val_idxs)
    np.random.shuffle(train_idxs)
    create_db(validation_db, train_image_mat_file, MAT_SHAPE, val_idxs, C, aug_data)
    create_db(train_db, train_image_mat_file, MAT_SHAPE, train_idxs, C, aug_data)
    create_db(test_db, test_image_mat_file, MAT_SHAPE, np.arange(MAT_SHAPE[0]), C, False)
    create_db(whole_train_db, train_image_mat_file, MAT_SHAPE, np.arange(MAT_SHAPE[0]), C, aug_data)
    print 'Done!'