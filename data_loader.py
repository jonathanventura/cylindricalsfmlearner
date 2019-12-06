from __future__ import division
import os
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None,
                 img_width=None,
                 num_source=None,
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'],
            seed=seed,
            shuffle=True)
        cam_paths_queue = tf.train.string_input_producer(
            file_list['cam_file_list'],
            seed=seed,
            shuffle=True)

        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])

        # Form training batches
        src_image_stack, tgt_image, intrinsics = \
                tf.train.batch([src_image_stack, tgt_image, intrinsics],
                               batch_size=self.batch_size)

        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        return tgt_image, src_image_stack, intrinsics

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, 0, tgt_start_idx, 0],
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0, 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, 0, int(tgt_start_idx + img_width), 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, 0, i*img_width, 0],
                                    [-1, -1, img_width, -1])
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
