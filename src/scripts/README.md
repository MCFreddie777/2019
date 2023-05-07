# Scripts

Helper scripts to generate or process data directly from console.
No need to run jupyter notebooks.

## Usage

From root directory of project run
```shell
docker compose exec jupyterlab python -m src.scripts.script_name
```

e.g.
```shell
docker compose exec jupyterlab python -m src.scripts.ground_truth
```
