# On-device AI Voice Assistant

Run `./install.sh` to setup environment and download models.

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
