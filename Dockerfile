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
    curl \
    unzip \
    gcc \
    python3 \
    python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Install pipenv
RUN pip3 install pipenv

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .

RUN PIPENV_VENV_IN_PROJECT=1 pipenv install

ENV PIPENV_CUSTOM_VENV_NAME="/.venv/"

WORKDIR /home

