FROM pure/python:3.8-cuda10.2-base
CMD nvidia-smi

# Locale
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Stop Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Enable Python tracebacks on segfaults
ENV PYTHONFAULTHANDLER 1

# https://github.com/NVI/nvidia-docker/issues/1632#issuecomment-1112667716
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Install OS dependencies
RUN set -xe && \
    apt-get update && \
    apt-get install  --no-install-recommends --no-install-suggests -y  \
    unzip

# Install conda & create the environment
ENV PATH="/root/miniconda3/bin:${PATH}"

COPY environment.yml .

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" \
    && conda env create -n trivago -f environment.yml \
    && conda init bash \
    && echo 'conda activate trivago' >> /root/.bashrc \
    && . /root/.bashrc \

# Override default shell and use bash
SHELL ["conda", "run", "--no-capture-output", "-n", "trivago", "/bin/bash", "-c"]

WORKDIR /home

