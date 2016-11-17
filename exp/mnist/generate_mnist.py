#!/usr/bin/env python
import os, os.path, gzip, re, sys, time, tempfile, shutil, struct, subprocess, argparse
import requests
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MNIST_TRAIN_IMAGES_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
MNIST_TEST_IMAGES_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
MNIST_TEST_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def error(error_message, error_code=1):
    print error_message
    sys.exit(error_code)

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
    contents = [new_file]
    with gzip.open(file_name, 'rb') as f:
        data = f.read()
        with open(new_file, 'wb') as fout:
            fout.write(data)
    return contents

class DBWrapper(object):
    @classmethod
    def supported_db_types(cls):
        return ['image_data']

    def __init__(self, db_type='lmdb'):
        self._db_type = db_type
        self._db_opened = False
        if db_type not in DBWrapper.supported_db_types():
            raise Exception('Unsupported db type: ' + db_type)
    
    def open(self, db_name, mode):
        if self._db_opened:
            raise Exception('DB already open!')
        self._db_opened = True
        if mode not in ['r', 'w']:
            raise Exception('Unsupported mode: ' + mode)
        self._db_name = db_name
        self._db_mode = mode
        self._db_index = os.path.join(db_name, 'index.txt')
        if mode == 'r':
            if not os.path.isdir(self._db_name) or not os.path.isfile(self._db_index):
                raise Exception("DB doens't exist!")
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
        pass
    def commit(self):
        pass
    def put(self, key, value):
        filename = os.path.join(self._db_name, '%s.png' % key)
        label, image = value
        cv2.imwrite(filename, image)
        with open(self._db_index, 'a') as f:
            if len(self._index_data) > 0:
                self._index_data += '\n'
                f.write('\n')
            record = '%s %d' % (os.path.basename(filename), label) 
            f.write(record)
            self._index_data += record


def read_data_to_db(images, labels, db_name, db_type='lmdb', from_idx=0, to_idx=-1):
    def read_image(f_img, f_lbl, rows, cols):
        label = struct.unpack('<B', f_lbl.read(1))[0]
        image_data = f_img.read(rows * cols)
        image = np.fromstring(image_data, dtype=np.uint8).reshape(rows, cols, 1)
        return (label, image)
    
    db = DBWrapper(db_type)
    try:
        db.open(db_name, 'w')
        with open(images, 'rb') as f_img:
            with open(labels, 'rb') as f_lbl:
                magic1 = struct.unpack('>i', f_img.read(4))[0]
                if magic1 != 2051:
                    print 'wrong magic!'
                    raise
                magic2 = struct.unpack('>i', f_lbl.read(4))[0]
                if magic2 != 2049:
                    print 'wrong magic!'
                    raise
                num_items = struct.unpack('>i', f_img.read(4))[0]
                num_labels = struct.unpack('>i', f_lbl.read(4))[0]
                if num_items != num_labels:
                    print 'Mismatch labels and items count!'
                    raise
                if to_idx < 0:
                    to_idx = num_items
                if from_idx > to_idx:
                    print 'From idx should be smaller than to idx.'
                    raise
                if from_idx < 0:
                    print 'from_idx should be non-negative.'
                    raise
                if to_idx > num_items:
                    print 'to_idx should be less than number of items.'
                    raise
                
                rows = struct.unpack('>i', f_img.read(4))[0]
                cols = struct.unpack('>i', f_img.read(4))[0]
                
                total_records = to_idx - from_idx
                char_length = len(str(total_records))
                key_template = '%%0%dd' % char_length
                record = 0
                for i in range(to_idx):
                    label, image = read_image(f_img, f_lbl, rows, cols)
                    if i < from_idx:
                        continue
                    db.put(key_template % record, (label, image))
                    record += 1
    finally:
        db.commit()
        db.close()

def create_dbs(zip_files, whole_train_db, train_db, validation_db, test_db, db_type):
    # Check if all the DBs exist
    if not filter(lambda x: not os.path.isdir(x), [train_db, whole_train_db, validation_db, test_db]):
        return
    files = []
    try:
        print '#######'
        dirname = os.path.dirname(zip_files[0][0])
        train_images = unarchive_data(zip_files[0][0], dirname)[0]
        train_labels = unarchive_data(zip_files[0][1], dirname)[0]
        test_images = unarchive_data(zip_files[1][0], dirname)[0]
        test_labels = unarchive_data(zip_files[1][1], dirname)[0]
        print '#######'
        if not os.path.isdir(whole_train_db):
            print 'Writing the whole train data...'
            read_data_to_db(train_images, train_labels, whole_train_db, db_type)
        else:
            print 'Whole train DB already exists.'
        print '#######'
        if not os.path.isdir(train_db):
            print 'Writing partial train data for cross-validation...'
            read_data_to_db(train_images, train_labels, train_db, db_type, from_idx=0, to_idx=50000)
        else:
            print 'Partial train DB already exists.'
        print '#######'
        if not os.path.isdir(validation_db):
            print 'Writing validation data...'
            read_data_to_db(train_images, train_labels, validation_db, db_type, from_idx=50000, to_idx=60000)
        else:
            print 'Validation DB already exists.'
        print '#######'
        if not os.path.isdir(test_db):
            print 'Writing test data...'
            read_data_to_db(test_images, test_labels, test_db, db_type)
        else:
            print 'Test DB already exists.'
    except:
        if os.path.isdir(train_db):
            shutil.rmtree(train_db)
        if os.path.isdir(validation_db):
            shutil.rmtree(validation_db)
        if os.path.isdir(test_db):
            shutil.rmtree(test_db)
        raise
    finally:
        for f in files:
            if os.path.isfile(f) or os.path.isdir(f):
                shutil.rmtree(f)

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--db_type",
        help="The type of database backend: %s [default = 'image_data']" % ', '.join(DBWrapper.supported_db_types()),
        default='image_data'
    )
    args = parser.parse_args()
    db_type = args.db_type
    whole_train_db = os.path.join(SCRIPT_DIR, 'data/mnist_whole_train_' + db_type)
    train_db = os.path.join(SCRIPT_DIR, 'data/mnist_train_' + db_type)
    validation_db = os.path.join(SCRIPT_DIR, 'data/mnist_validation_' + db_type)
    test_db = os.path.join(SCRIPT_DIR, 'data/mnist_test_' + db_type)
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
    train_img_tar_file = download_data(MNIST_TRAIN_IMAGES_URL, os.path.join(SCRIPT_DIR, 'data'))
    train_label_tar_file = download_data(MNIST_TRAIN_LABELS_URL, os.path.join(SCRIPT_DIR, 'data'))
    train = [train_img_tar_file, train_label_tar_file]
    test_img_tar_file = download_data(MNIST_TEST_IMAGES_URL, os.path.join(SCRIPT_DIR, 'data'))
    test_label_tar_file = download_data(MNIST_TEST_LABELS_URL, os.path.join(SCRIPT_DIR, 'data'))
    test = [test_img_tar_file, test_label_tar_file]
    # Create train and test DBs
    create_dbs([train, test], whole_train_db, train_db, validation_db, test_db, db_type)
    print '#######'
    print 'Done!'
    
if __name__ == '__main__':
    main(sys.argv)
