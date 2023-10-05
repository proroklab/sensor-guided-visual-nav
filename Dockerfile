FROM nvidia/cudagl:11.1-devel-ubuntu20.04

# Fix NVIDIA CUDA Linux repository key rotation
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Disable dpkg/gdebi interactive dialogs
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git python3 python3-pip nano lsof

# Make image smaller by not caching downloaded pip pkgs
ARG PIP_NO_CACHE_DIR=1

# From https://github.com/cyberbotics/webots-docker/blob/master/Dockerfile:
# Install Webots runtime dependencies
RUN apt update && apt install --yes wget && rm -rf /var/lib/apt/lists/
ARG WEBOTS_VERSION=R2023a
RUN wget https://raw.githubusercontent.com/cyberbotics/webots/master/scripts/install/linux_runtime_dependencies.sh
RUN chmod +x linux_runtime_dependencies.sh && ./linux_runtime_dependencies.sh && rm ./linux_runtime_dependencies.sh && rm -rf /var/lib/apt/lists/
# Install X virtual framebuffer to be able to use Webots without GPU and GUI (e.g. CI)
RUN apt update && apt install --yes xvfb && rm -rf /var/lib/apt/lists/

# Install Webots
WORKDIR /usr/local
RUN wget https://github.com/cyberbotics/webots/releases/download/$WEBOTS_VERSION/webots-$WEBOTS_VERSION-x86-64.tar.bz2
RUN tar xjf webots-*.tar.bz2 && rm webots-*.tar.bz2
ENV QTWEBENGINE_DISABLE_SANDBOX=1
ENV WEBOTS_HOME /usr/local/webots
ENV PATH /usr/local/webots:${PATH}

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH $CONDA_DIR/bin:$PATH

# Configure conda to use libmamba for faster environment resolve
RUN conda init bash
RUN conda install -n base conda-libmamba-solver && conda config --set solver libmamba

WORKDIR /tmp
ADD environment.yml .
RUN conda env create -f environment.yml

WORKDIR /opt/sensor-guided-visual-nav

# Make webots available for external controller
ENV WEBOTS_HOME /usr/local/webots
ENV PYTHONPATH $PYTHONPATH:${WEBOTS_HOME}/lib/controller/python
ENV PYTHONIOENCODING UTF-8

#RUN useradd -ms /bin/bash vnav

#USER vnav
#WORKDIR /home/vnav
