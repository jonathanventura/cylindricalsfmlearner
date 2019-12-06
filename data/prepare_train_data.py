from __future__ import division
import argparse
import cv2
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os

# Command-line Interface

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Where the dataset is stored")
parser.add_argument("--dataset_name", type=str, required=True,
                    choices=["headcam"])
parser.add_argument("--dump_root", type=str, required=True,
                    help="Where to dump the data")
parser.add_argument("--seq_length", type=int, required=True,
                    help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128,
                    help="Image height")
parser.add_argument("--img_width", type=int, default=416,
                    help="Image width")
parser.add_argument("--num_threads", type=int, default=4,
                    help="Number of threads to use")
parser.add_argument("--val_frac", type=float, default=0.1,
                    help="Fraction of data to use for validation")
args = parser.parse_args()

# Helpers

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n, dump_root):
    """Dumps nth example (+intrinsics) to formatted files."""
    if n % 200 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))

    try:
        example = data_loader.get_train_example_with_idx(n)
        if example == False:
            return
    except:
        print('bad image')
        return

    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    f_theta = intrinsics[0]
    c_theta = intrinsics[1]
    f_Z = intrinsics[2]
    c_Z = intrinsics[3]

    dump_dir = os.path.join(dump_root, example['folder_name'])
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise

    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    cv2.imwrite(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (f_theta, c_theta, f_Z, c_Z))

# Main

def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == 'headcam':
        from headcam.headcam_loader import headcam_loader
        data_loader = headcam_loader(args.dataset_dir,
                                     img_height = args.img_height,
                                     img_width=args.img_width,
                                     seq_length=args.seq_length)

    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args.dump_root) \
        for n in range(data_loader.num_train))

    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    trainfile = os.path.join(args.dump_root, 'train.txt')
    valfile = os.path.join(args.dump_root, 'val.txt')
    with open(trainfile, 'w') as tf:
        with open(valfile, 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in sorted(frame_ids):
                    if np.random.random() < args.val_frac:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))

main()
