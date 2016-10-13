#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess, sys, os, argparse, inspect, importlib
SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
UTILS_DIR = os.path.abspath(SCRIPT_DIR)
sys.path.append(UTILS_DIR)
import generate_missing_data as missing_data
def main(argv):
    parser = missing_data.get_base_argparser(argv)
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Relative path (from working directory) to the dataset directory."
    )
    args = parser.parse_args()
    while args.dataset_dir[-1] == '/':
        args.dataset_dir = args.dataset_dir[:-1]
    description = []
    if args.MAR_prob > 0:
        description.append('mar_%.2f' % args.MAR_prob)
    if args.max_rects > 0 and args.max_width > 0:
        description.append('minrects_%d' % args.min_rects)
        description.append('maxrects_%d' % args.max_rects)
        description.append('minwidth_%d' % args.min_width)
        description.append('maxwidth_%d' % args.max_width)
    if len(description) == 0:
        print 'No corruption arguments supplied!'
        sys.exit(0)
    description_str = '_'.join(description)
    args.output_dir = '%s_%s' % (args.dataset_dir, description_str)
    if os.path.isdir(args.output_dir):
        print 'Dataset already exists!'
        sys.exit(0)
    original_index = None
    with open('%s/index.txt' % args.dataset_dir, 'r') as f:
        original_index = [(' '.join(line.strip().split(' ')[:-1]), line.strip().split(' ')[-1]) for line in f if len(line.strip()) > 0]
    args.filenames = ['%s/%s' % (args.dataset_dir, filename) for filename, _ in original_index]
    try:
        missing_data.convert_dataset(args)
    except KeyboardInterrupt:
        print 'User stopped generation!'
        sys.exit(0)
    with open('%s/index.txt' % args.output_dir, 'w') as f:
        for filename, c in original_index:
            f.write('%s %s\n' % (filename[:-4] + '_corrupted.png', c))
    with open('%s/index_mask.txt' % args.output_dir, 'w') as f:
        for filename, c in original_index:
            f.write('%s %s\n' % (filename[:-4] + '_mask.png', c))

if __name__ == '__main__':
    main(sys.argv)

