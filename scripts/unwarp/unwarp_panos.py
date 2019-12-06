import numpy as np
import os
import sys
from scipy.io import loadmat
from scipy.misc import imread, imsave
import tensorflow as tf
from tqdm import trange

# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import bilinear_sampler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def unwarp_panos(infiles, outdir, lutfile='data/lut.npy', debug=False):
    """
    Unwarps the given files from spherical to cylindrical projection.
    """
    lut = np.load(lutfile)
    lutx = lut[:,:,0].astype('float32')
    luty = lut[:,:,1].astype('float32')

    # get configuration

    im = imread(infiles[0])

    im_ph = tf.placeholder('uint8',im.shape)
    imgs = tf.cast(im_ph,'float32')
    imgs = tf.expand_dims(imgs,axis=0)

    lutx_ph = tf.placeholder('float32',lutx.shape)
    luty_ph = tf.placeholder('float32',luty.shape)

    lutx_re = tf.expand_dims(lutx_ph,axis=0)
    lutx_re = tf.expand_dims(lutx_re,axis=-1)
    luty_re = tf.expand_dims(luty_ph,axis=0)
    luty_re = tf.expand_dims(luty_re,axis=-1)

    coords = tf.concat([lutx_re,luty_re],axis=-1)
    pano = bilinear_sampler(imgs,coords)

    # unwarp all files

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        for i in trange(len(infiles), desc='Unwarping panorama'):
            infile = infiles[i]

            # set the outfile
            _, filename = os.path.split(infile)
            outfile = os.path.join(outdir, filename)

            # run the unwarp
            im = imread(infile)
            res = sess.run(pano, {im_ph:im,lutx_ph:lutx,luty_ph:luty})
            res = np.squeeze(res, axis=0)
            res = res.astype('uint8')

            # save
            imsave(outfile, res)
            if debug:
                print('Saved {}.'.format(outfile))
