#!/usr/bin/env python
import numpy as np
from numpy.random import random_sample, random_integers
import scipy.ndimage as nd
import PIL.Image
import os, os.path, sys, argparse, errno
import progressbar
from multiprocessing import Pool
import math
import signal

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    np.random.seed(None)

# Utilities for loading and saving images to/from numpy arrays
def load_image(path, scale=255.0):
    return np.float32(PIL.Image.open(path)) / scale
def save_image(img, path, fmt='png', scale=255.0, quality=100):
    if img.dtype != np.uint8:
        img = np.uint8(np.clip(img*scale, 0, 255))
    if len(os.path.splitext(path)[-1]) > 0:
        fmt = os.path.splitext(path)[-1][1:]
    else:
        path = '%s.%s' % (path, fmt)
    PIL.Image.fromarray(img).save(path, fmt, quality=quality)
'''
Corrupt an image by two methods:
    1) Missing At Random (MAR) - remove a pixel with probability MAR_prob.
    2) Remove random rectangles - min/max_rects control the number of rectangles,
       and min/max_width control the size of the rectangles.
'''
def corrupt_image(img, MAR_prob=0, min_rects=0, max_rects=0, min_width=0, max_width=0, apply_to_all_channels=False):
    def generate_channel_mask():
        mask = np.zeros(img.shape[0:2], dtype=np.bool)
        if MAR_prob > 0:
            mask[(random_sample(mask.shape) < MAR_prob)] = True
        if max_rects > 0 and max_width > 0:
            h, w = mask.shape
            num_rects = random_integers(min_rects, max_rects)
            for i in range(num_rects):
                px1 = random_integers(0, w - min(max(min_width, 1), w))
                py1 = random_integers(0, h - min(max(min_width, 1), h))
                px2 = px1 + min_width + random_integers(0, max(min(w - px1 - min_width, max_width - min_width), 0));
                py2 = py1 + min_width + random_integers(0, max(min(h - py1 - min_width, max_width - min_width), 0));
                if px1 <= px2 and py1 <= py2:
                    mask[py1:py2, px1:px2] = True
                else:
                    # One of the sides has length 0, so we should remove any pixels4
                    pass
        return mask
    new_img = img.copy()
    channels = 1 if len(new_img.shape) == 2 else new_img.shape[-1]
    global_mask = np.zeros(img.shape, dtype=np.bool)
    if channels == 1 or apply_to_all_channels:
        mask = generate_channel_mask()
        if channels == 1:
            global_mask[:, :] = mask
        else:
            for i in xrange(channels):
                global_mask[:, :, i] = mask
    else:
        global_mask = np.zeros(img.shape, dtype=np.bool)
        for i in xrange(channels):
            global_mask[:,:,i] = generate_channel_mask()
    new_img[global_mask] = 0
    return (new_img, 1.0 * global_mask)

# Process command line inputs
def get_base_argparser(argv):
    def positive_integer(x):
        x = int(x)
        if x < 0:
            raise ValueError('parameter must be a positive integer')
        return x
    def probability(x):
        x = float(x)
        if x < 0 or x > 1:
            raise ValueError('parameter must be a real number between 0 and 1 (inclusive)')
        return x
    def bool_value(x):
        if x == 'y':
            return True
        elif x == 'n':
            return False
        else:
            raise ValueError("parameter must be either 'y' or 'n', representing yes and no respectively.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MAR_prob",
        type=probability,
        default=0.0,
        help="Missing At Random (MAR) - the probability of removing single pixels from the image."
    )
    parser.add_argument(
        "--min_rects",
        type=positive_integer,
        default=0,
        help="minimum number of squares that will be removed from the image."
    )
    parser.add_argument(
        "--max_rects",
        type=positive_integer,
        default=0,
        help="maximum number of squares that will be removed from the image."
    )
    parser.add_argument(
        "--min_width",
        type=positive_integer,
        default=0,
        help="minimum width of the squares that will be removed from the image."
    )
    parser.add_argument(
        "--max_width",
        type=positive_integer,
        default=0,
        help="maximum width of the squares that will be removed from the image."
    )
    parser.add_argument(
        "--same_mask_to_all_channels",
        type=bool_value,
        default=False,
        help="For color images, if true ('y') then the same mask will be applied to all channels. Otherwise a different mask will be used for each channel."
    )
    return parser 
def get_args(argv):
    parser = get_base_argparser(argv)
    # Required arguments: input and output files.
    parser.add_argument(
        "output_dir",
        type=str,
        help="output dir for the corrupted and mask images. New filenames are based on '%%s_corrupted.png' or '%%s_mask.png'"
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs='+',
        help="a space seperated list of image filenames that the program will corrupt"
    )
    return parser.parse_args()

def corrupt_source_image(params):
    source_path, args = params
    img = load_image(source_path)
    corrupted_img, mask = corrupt_image(img,
        MAR_prob=args.MAR_prob,
        min_rects=args.min_rects,
        max_rects=args.max_rects,
        min_width=args.min_width,
        max_width=args.max_width,
        apply_to_all_channels=args.same_mask_to_all_channels)
    filename = os.path.splitext(os.path.basename(source_path))[0]
    save_image(corrupted_img, '%s/%s_corrupted.png' % (args.output_dir, filename))
    save_image(mask, '%s/%s_mask.png' % (args.output_dir, filename))
    return True

def convert_dataset(args):
    try:
        if args.min_rects > args.max_rects:
            raise ValueError('min_rect must be less than or equal to max_rect.')
        if args.min_width > args.max_width:
            raise ValueError('min_width must be less than or equal to max_width.')
        try:
            os.makedirs(args.output_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(args.output_dir):
                pass
            else:
                raise ValueError('output_dir argument is not a valid path.')
        total_images = len(args.filenames)
        params = zip(args.filenames, [args] * total_images)
        pool = Pool(initializer=init_worker)
        pbar = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('\rProcessed %(value)d of %(max)d Images '), progressbar.Bar()], maxval=total_images, term_width=50).start()
        try:
            results = pool.imap_unordered(corrupt_source_image, params, chunksize=max(int(math.sqrt(len(args.filenames)))/2, 10))
            for i in range(len(args.filenames)):
                next(results)
                pbar.update(i+1)
            pool.close()
            pool.join()
            pbar.finish()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pbar.finish()
            raise
    except ValueError as e:
        print
        print 'Bad parameters:', e
        raise e
    except KeyboardInterrupt:
        print
        if __name__ == '__main__':
            print 'User stopped generation!'
        raise 
    except:
        print
        print "Unexpected error:", sys.exc_info()[0]
        raise
# Main routine
def main(argv):
    args = get_args(argv)
    convert_dataset(args)
if __name__ == '__main__':
    main(sys.argv)
