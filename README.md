## Unsupervised learning of depth and ego-motion from cylindrical panoramic video

This repository hosts the code for our paper.

Alisha Sharma and Jonathan Ventura.  "Unsupervised Learning of Depth and Ego-Motion from Cylindrical Panoramic Video."  Proceedings of the 2019 IEEE Artificial Intelligence & Virtual Reality Conference, San Diego, CA, 2019.

If you make use of this code or dataset, please cite this paper.

The code is based on [SfMLearner by Zhou et al](https://github.com/tinghuiz/SfMLearner).

### Abstract

We introduce a convolutional neural network model for unsupervised learning of depth and ego-motion from cylindrical panoramic video. Panoramic depth estimation is an important technology for applications such as virtual reality, 3d modeling, and autonomous robotic navigation. In contrast to previous approaches for applying convolutional neural networks to panoramic imagery, we use the cylindrical panoramic projection which allows for the use of the traditional CNN layers such as convolutional filters and max pooling without modification. Our evaluation of synthetic and real data shows that unsupervised learning of depth and ego-motion on cylindrical panoramic images can produce high-quality depth maps and that an increased field-of-view improves ego-motion estimation accuracy. We also introduce Headcam, a novel dataset of panoramic video collected from a helmet-mounted camera while biking in an urban setting.

This paper won the best paper award at IEEE AIVR 2019!

### Dataset

Our Headcam dataset is [hosted on Zenodo](https://zenodo.org/record/3520963).

This dataset contains panoramic video captured from a helmet-mounted camera while riding a bike through suburban Northern Virginia.  We used the videos to evaluate an unsupervised learning method for depth and ego-motion estimation, as described in our paper.
 
The videos are stored as .mkv video files encoded using lossless H.264.  To extract the images, we recommend using ffmpeg:

    mkdir 2018-10-03 ;

    ffmpeg -i 2018-10-03.mkv -q:v 1 2018-10-03/%05d.png ;
    
### Data preparation

    python data/prepare_training_data.py --dataset_dir <path-to-headcam> --dataset_name 'headcam' --dump_root <output-path> --seq_length 3 --img_height 512 --img_width 2048 --val_frac 0.1
    
### Training

    python train.py --dataset_dir <path-to-prepared-data> --img_height 512 --img_width 2048
    
### Testing

