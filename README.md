# [DP] Session based recommendation system

## Prerequisites

Getting Started

- Make sure you have installed the NVIDIA driver installed on the host system
- [Docker](https://docs.docker.com/get-docker/)
- Download [dataset](https://www.kaggle.com/pranavmahajan725/trivagorecsyschallengedata2019) into `/data` directory.

## Spawning and running the container

```
docker compose up
```

## Usage

Go to [http://127.0.0.1:8888](http://127.0.0.1:8888) in your browser or click the link in the console.


## Installing packages

This image contains `conda` + `pip`, therefore install the packages as followed:
```
docker compose exec jupyterlab conda install <package>
```

or

```
docker compose exec jupyterlab pip install <package>
```

To persist the packages when rebuilding the image, packages need to be freezed into environment file.

```
docker compose exec jupyterlab conda env export --no-build > environment.yml
```

## Scripts

Projectr also contains helper scripts to generate or process data directly from console.
No need to run jupyter notebooks.

### Usage

From root directory of project run
```shell
docker compose exec jupyterlab python src/<script_name>.py
```

e.g.
```shell
docker compose exec jupyterlab python src/preprocess.py
```
