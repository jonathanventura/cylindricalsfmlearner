FROM tensorflow/tensorflow:1.11.0-gpu-py3

WORKDIR /

RUN apt-get update ; apt-get install vim git curl unzip ffmpeg wget llvm-6.0 freeglut3 freeglut3-dev -y ; 

RUN pip install --upgrade scikit-image tqdm keras scipy requests plyfile opencv-python-headless ;

RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb ; \
    dpkg -i ./mesa_18.3.3-0.deb || true ; \
    apt install -f -y ; 

RUN git clone https://github.com/mmatl/pyopengl ; \
    pip install ./pyopengl

RUN git clone https://github.com/mikedh/trimesh.git ; \
    pip install ./trimesh

RUN pip install --upgrade pyrender

RUN rm /usr/bin/python; ln -s /usr/bin/python3.5 /usr/bin/python


