FROM leesharma/cylindricalsfm:gpu
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

