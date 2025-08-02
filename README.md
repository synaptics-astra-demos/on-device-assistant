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

### Additional Demos
Model-specific examples are available in the [demos/](demos/) directory:
```sh
source .venv/bin/activate
python -m demos.<model>
```

## Profiling
Profiling scripts are available at [profile/](profile/). Currently supported models:

| Model | Script | Supported Model Types |
| ----- | ------ | ----------------- |
| MiniLM | [profile/minilm.py](profile/minilm.py) | `.synap` (SyNAP), `.gguf` (llama.cpp) |
| Moonshine | [profile/moonshine.py](profile/moonshine.py) | `.synap` (SyNAP), `.onnx` (ORT) |

Run profiling with:
```sh
source .venv/bin/activate
python -m profile.<model>
```

#### Run Options
* `--model[s]`: Model(s) to profile, inference runner is selected based on model type
* `--run-forever`: Continuosly profile provided models in a loop until interrupted with `ctrl + c`
* `-j`: Number of cores to use for CPU execution (default: all)

> [!TIP]
> Use in conjunction with the [Astra resource usage visualizer](https://github.com/spal-synaptics/astra-visualizer) to get a live dashboard of CPU and NPU usage during inference
