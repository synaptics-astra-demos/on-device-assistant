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
* `--qa-file`: Path to Question-Answer pairs (default: [data/qa_dishwasher.json](data/qa_dishwasher.json))
* `-j`: Number of cores to use for CPU execution (default: all)
* `--cpu-only`: Run all models on the CPU (default for SL1620)

Run `python assistant.py --help` to view all available options

### Additional Demos
Model-specific examples are available in the [demos/](demos/) directory:
```sh
source .venv/bin/activate
python -m demos.<model>
```

## Profiling
Profiling scripts are available at [profile_model/](profile_model/). Currently supported models:

| Model | Script | Supported Model Types |
| ----- | ------ | ----------------- |
| MiniLM | [profile_model/minilm.py](profile_model/minilm.py) | `.synap` (SyNAP), `.gguf` (llama.cpp) |
| Moonshine | [profile_model/moonshine.py](profile_model/moonshine.py) | `.synap` (SyNAP), `.onnx` (ORT) |
| Opus-MT | [profile_model/opus_mt.py](profile_model/opus_mt.py) | `.synap` (SyNAP) [^1] |

Run profiling with:
```sh
source .venv/bin/activate
python -m profile_model.<model>
```

#### Run Options
* `--models`: Model(s) to profile, inference runner is selected based on model type
* `--run-forever`: Continuosly profile provided models in a loop until interrupted with `ctrl + c`
* `-j`: Number of cores to use for CPU execution (default: all)

> [!TIP]
> Use in conjunction with the [Astra resource usage visualizer](https://github.com/spal-synaptics/astra-visualizer) to get a live dashboard of CPU and NPU usage during inference

[^1]: See [Astra Opus-MT models](https://huggingface.co/collections/Synaptics/astra-sl-translation-models-683cb9bdb74ebbceba6cc55c) on HuggingFace for ONNX inference
