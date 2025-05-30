# On-device AI Voice Assistant

## Setup
Run `./install.sh` to setup environment and download models.

## Demo
Launch assistant with:
```sh
source .venv/bin/activate
python assistant.py
```

#### Run Options
* `--qa-file`: Path to Question-Answer pairs (default: [data/qa_assistant.json](data/qa_assistant.json))
* `--cpu-only`: Use CPU only models
* `-j`: Number of cores to use for CPU execution (default: all)

Run `python assistant.py --help` to view all available options

## Profiling
Profiling scripts are available at [profile/](profile/). Currently supported models:
* MiniLM [profile/minilm.py](profile/minilm.py)

Run profiling with:
```sh
source .venv/bin/activate
python -m profile.<model>
```

#### Run Options
* `--models`: Model files to profile, inference runner is selected based on model type
* `--run-forever`: Continuosly profile provided models in a loop until interrupted with `ctrl + c`
* `-j`: Number of cores to use for CPU execution (default: all)
