# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# tensorflow    latest (pip)
# ==================================================================

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        git \
        vim \
        wget \
        && \
# ==================================================================
# python
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        Pillow==5.3.0 \
        h5py==2.8.0 \
        jupyter==1.0.0 \
        matplotlib==3.0.0 \
        numpy==1.15.2 \
        pandas==0.23.4 \
        scipy==1.1.0 \
        scikit-learn==0.20.0 \
        opencv-python-headless==1.15.2 \
        joblib==0.12.5 \
        tqdm==4.26.0 \
        && \
# ==================================================================
# tensorflow
# ------------------------------------------------------------------
    $PIP_INSTALL \
        tensorflow==1.11.0 \
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN apt-get update

RUN apt-get install vim -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN pip install scikit-image tqdm keras
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
RUN apt-get install unzip -y
RUN curl https://rclone.org/install.sh | bash
RUN apt-get install unrar -y
RUN pip install --upgrade scipy
RUN apt-get install ffmpeg -y
RUN pip install requests
RUN pip install plyfile

WORKDIR /

RUN apt-get install llvm-6.0 freeglut3 freeglut3-dev -y ;
RUN apt-get install wget -y ;
RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb ;
RUN apt update ; \
    dpkg -i ./mesa_18.3.3-0.deb || true ; \
    apt install -f -y ;

RUN git clone https://github.com/mmatl/pyopengl ;\
    pip install ./pyopengl

RUN git clone https://github.com/mikedh/trimesh.git ;\
    pip install ./trimesh

RUN pip install --upgrade pyrender

