from __future__ import division
import numpy as np
from glob import glob
import os
import cv2


class headcam_loader(object):
    """Loads and processes Headcam data."""
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=512,
                 seq_length=3):
        """Initialize a new headcam data loader."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        excluded_frames_file = 'data/headcam/excluded_frames_5fps.txt'
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.date_list = [
            '2018-10-02',
            #'2018-10-03',
            '2018-10-07',
        ]
        self.collect_excluded_frames(excluded_frames_file)
        self.collect_train_frames()

    def collect_excluded_frames(self, excluded_frames_file):
        with open(excluded_frames_file, 'r') as f:
            frames = f.readlines()
        self.excluded_frames = []
        for fr in frames:
            if fr == '\n':
                continue
            subset, frame_id = fr.split(' ')
            curr_fid = '%.5d' % (np.int(frame_id[:-1]))
            self.excluded_frames.append(subset + ' ' + curr_fid)

    def collect_train_frames(self):
        """Collect all training frame files in a list."""
        all_frames = []
        for date in self.date_list:
            date_dir = os.path.join(self.dataset_dir, date)
            if os.path.isdir(date_dir):
                N = len(glob(date_dir + '/*.png'))
                for n in range(N):
                    frame_id = '%.05d' % n
                    all_frames.append('{} {}'.format(date_dir, frame_id))

        for excluded in self.excluded_frames:
            try:
                all_frames.remove(excluded)
                print('removed excluded frame from training: %s' % excluded)
            except:
                pass

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        """Checks if a frame can create a valid sample.

        This depends on the sequence length. For example, for a sequence
        length of 5, a frame with index `n` is valid if the sequence
        [n-2, n-1, n, n+1, n+2] exists.
        """
        N = len(frames)

        # drive location, picture id
        tgt_drive, _ = frames[tgt_idx].split(' ')

        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False

        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')

        return tgt_drive == min_src_drive and tgt_drive == max_src_drive

    def get_train_example_with_idx(self, tgt_idx):
        """Loads the training example.

        If the target is an invalid sample, returns False.
        """
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False

        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        """Loads the frames of an image sequence."""
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image_raw(curr_drive, curr_frame_id)
            # resize images
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = cv2.resize(curr_img,
                                           (self.img_width, self.img_height),interpolation=cv2.INTER_AREA)
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        """Loads the frames of an image sequence."""
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames,
                                                             tgt_idx,
                                                             self.seq_length)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics_raw()
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)

        # strip trailing slash (for basename)
        if tgt_drive[-1] == '/':
            tgt_drive = tgt_drive[:-1]

        # build example
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = os.path.basename(tgt_drive)
        example['file_name'] = tgt_frame_id
        return example

    def load_image_raw(self, drive, frame_id):
        """Returns a raw image given a drive and frame id."""
        date = drive[-10:]
        img_file = os.path.join(self.dataset_dir,
                                date,
                                drive,
                                frame_id + '.png')
        img = cv2.imread(img_file)
        return img

    def load_intrinsics_raw(self, intrinsics_path='./data/headcam/intrinsics.txt'):
        """Loads and returns unscaled headcam intrinsics."""
        with open(intrinsics_path, 'r') as f:
            line = f.readline().rstrip()

            # load theta and Z intrinsics
            f_theta, c_theta, f_Z, c_Z = line.split(' ')

            intrinsics = np.array([float(f_theta),
                                   float(c_theta),
                                   float(f_Z),
                                   float(c_Z)])
            f.close()
        return intrinsics

    def scale_intrinsics(self, mat, s_theta, s_Z):
        """Scale intrinsics for the given image scaling."""
        out = np.copy(mat)
        out[0] *= s_theta  # f_theta
        out[1] *= s_theta  # c_theta
        out[2] *= s_Z      # f_Z
        out[3] *= s_Z      # c_Z
        return out
