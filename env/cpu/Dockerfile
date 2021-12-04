FROM cschranz/gpu-jupyter:v1.4_cuda-11.0_ubuntu-20.04


USER root

ENV DEBIAN_FRONTEND noninteractive

# Install packages needed 
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        g++  \
        git  \
        curl  \
        cmake \
        zlib1g-dev \
        libjpeg-dev \
        xvfb \
        xorg-dev \
        libboost-all-dev \
        libsdl2-dev \
        swig \
        python3  \
        python3-dev  \
        python3-future  \
        python3-pip  \
        python3-setuptools  \
        python3-wheel  \
        python3-tk \
        libopenblas-base  \
        cython3  \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

RUN pip install \
    zipcodes \
    pymongo \
    dask-mongo \
    pytz \
    tqdm \
    numba \
    atari_py \
    gym[atari] \
    redis \
    walrus \
    msgpack_numpy
    


COPY ./lib/roms /home/jovyan/lib/roms

RUN python -m atari_py.import_roms /home/jovyan/lib/roms 

ENV PYTHONPATH=.:/home/jovyan/working
ENV TF_CPP_MIN_LOG_LEVEL=2

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
