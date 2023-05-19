# [DP] Session based recommendation system

## Prerequisites

Getting Started

- Make sure you have installed the NVIDIA driver installed on the host system
- [Docker](https://docs.docker.com/get-docker/)
- Download [dataset](https://recsys2019data.trivago.com/) into `/data/input` directory.

## Spawning and running the container

```
docker compose up
```

## Usage

Go to [http://127.0.0.1:8888](http://127.0.0.1:8888) in your browser or click the link in the console.


## Installing packages

This image contains `conda` + `pip`, therefore install the packages as followed:
```
docker compose exec jupyterlab bash -ic "conda install <package>"
```

or

```
docker compose exec jupyterlab bash -ic "pip install <package>"
```

To persist the packages when rebuilding the image, packages need to be freezed into environment file.

```
docker compose exec jupyterlab bash -ic "conda env export --no-build > environment.yml"
```

## Scripts

Project also contains helper scripts to generate or process data directly from console.
No need to run jupyter notebooks.

### Usage

1. Clone `.env.example` file and rename it to `.env` file. Change the env variables if you wish or keep the defaults.

1. From root directory of project run
```
docker compose exec jupyterlab bash -ic "python src/<script_name>.py"
```

e.g.
```shell
docker compose exec jupyterlab bash -ic "python src/preprocess.py"
```
