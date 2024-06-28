# Create a working directory
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe
ARG HOST_UNAME
ARG HOST_UID
ARG HOST_GID


# Install some basic utilities
RUN rm -f /etc/apt/sources.list.d/*.list \
    && rm -vf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /deps

COPY docker/inputrc /root/.inputrc

SHELL ["/bin/bash", "-c"]

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
    
ENV PATH=/opt/conda/bin:$PATH


# bootstrap conda
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate base \
    && conda update -n base conda -y \
    && conda install python=3.8 \
    && python --version \
    && source /opt/conda/etc/profile.d/conda.sh \
    && conda init bash \
    && conda config --set auto_activate_base true

ENV FORCE_CUDA="1"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# We use cuda-11.1 by default
RUN conda install cudatoolkit=11.1.1 -c conda-forge --yes \
&& pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install python packages
RUN pip3 install -U pip && pip3 install -U setuptools==59.5.0

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate base \
    && conda install -c conda-forge \
        matplotlib \
        tqdm \
        flatten-dict \
        imageio \
        joblib \
        numba \
        numpy \
        pandas \
        psutil \
        pyyaml \
        scipy \
        click \
        scikit-optimize \
        plotly \
        seaborn \
        tensorboard \
    && pip install datadiff \
        tiktoken

# add user of current host system
RUN groupadd -g ${HOST_GID} -o ${HOST_UNAME}
RUN useradd -l -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_UNAME}

# make sure conda env is always active on startup there as well
USER ${HOST_UNAME}


RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda init bash \
    && conda config --set auto_activate_base true

# switch back to root as default user
USER root

# copy inputrc to user for a nicer bash experience
RUN cp /root/.inputrc /home/${HOST_UNAME}/.inputrc \
    && chown ${HOST_UNAME}:${HOST_GID} /home/${HOST_UNAME}/.inputrc

WORKDIR /workspace
