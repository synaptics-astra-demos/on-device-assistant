# On-device AI Voice Assistant

## Setup
Run `./install.sh` to setup environment and download models.

> Please note this Demo code is compatible with  Astra OOBE SDK 1.8 (kirkstone) and below.Images.


This demonstration showcases an contextual AI voice assistant operating entirely on-device, with no cloud dependency or offloading, designed to run on [Synaptics Astra](https://www.synaptics.com/products/embedded-processors/astra-machina-foundation-series) Machina board. 

- Understanding context-specific queries.
- Responding quickly (as low as 500ms) and accurately without cloud dependence.
- Tool calling peripherals or cloud services for extended functionality.
- Leveraging multi-modal systems for vision-based queries.
- Hallucination-free responses, providing Q&A dataset is clean (see Limitations).

This project builds upon the work and contributions of many open source AI projects and individuals, including:

1. **Speech-to-Text**: [Moonshine](https://github.com/usefulsensors/moonshine) SyNAP model running on NPU, which is 5x faster than Whisper with better accuracy.
2. **Response Generation**: Context-specific Q&A matching using an encoder-only [language model](https://www.sbert.net/docs/quickstart.html) (future support for small LLM planned).
3. **Text-to-Speech**: [Piper](https://github.com/rhasspy/piper) by the Open Home Foundation.
4. **Voice Activity Detection**: [Silero VAD](https://github.com.mcas.ms/snakers4/silero-vad) pre-trained model to filter out humming and similar non-speech sounds and end of utterance.

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
