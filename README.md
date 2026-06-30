# Runtime Environment

## Python Version
- Python **3.10.xx** for compatibility (handled by [**Miniconda**](#using-conda-to-create-the-environment)).

## Choosing the right environment file

⚠️ **Important**: Ensure that you have the latest driver version installed (preferrably Studio for stability) to ensure the CUDA toolkit is compatible with your GPU architecture.

1. **(Recommended)** _Checking for driver specific CUDA version:_

   ```bash
   $ nvidia-smi

   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.8     |
   |-------------------------------+----------------------+----------------------+
   ```

2. _Minimum architectural requirements:_

   **For Turing (RTX 20 Series):**
   - CUDA 10.0 or later is required.

   **For Ampere (RTX 30 Series):**
   - CUDA 11.0 or later is required.

   **For Ada Lovelace (RTX 40 Series):**
   - CUDA 12.0 or later is required.

   **For Blackwell (RTX 50 Series):**
   - CUDA 12.8 or later is required.

3. **(Not Recommended)** Running on CPU
   - It's possible but will be significantly slower. If you choose this option, use the `environment.yml` file.

## Using Conda to create the environment
1. Ensure you have Miniconda installed on your device. If you don't, you can get the latest version from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).
2. Open any terminal instance and run the following command:
   
   ```bash
   $ conda env create -f path/to/file.yml
   ```
3. Environment will be created with the name `maestro-cudaXXX` where `XXX` is the CUDA version mentioned in the file name, or alternatively `maestro` if the default `environment.yml` file is chosen. To activate the environment simply run:
   ```bash
   $ conda activate env_name
   ```
   
   You can also directly select the environment in **Visual Studio Code** through the `Select Interpreter` option found in the bottom right corner of your editor window.
4. To update the current environment after modifying the YAML file in future versions, simply run:
   ```bash
   $ conda activate env_name
   $ conda env update -f path/to/file.yml --prune
   ```

## Software Requirements
1. [**Chocolatey**](https://chocolatey.org/install) package manager  for installing **FluidSynth**:
   ### Installing Chocolatey using PowerShell:
      ```powershell
      $ Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
      ```

2. [**FluidSynth**](https://www.fluidsynth.org/wiki/Download) for using Soundfonts in *[AudioEngine](./src/core/audio_engine.py)*'s live audio generation.
   ### Installing FluidSynth using Chocolatey:
   ```bash
   $ choco install fluidsynth
   ```
   

# Generating Music
## a. Running the Inference Server (recommended)

To run the inference WebSocket server, using our best model:


```bash
$ uvicorn src.core.inference_ws_server:app --host 127.0.0.1 --port 8000
```

## b. Generating music directly from the command line
To generate tokens without the need to run the Inference server, you can call individual models with custom command-line arguments:

```bash
$ python -m src.models.model_name generate --command-1 argument-1 --command-2 argument-2
```
### Available Model Names (sorted by generative performance):
1. [chrollo_0](/src/models/chrollo.py#L80)
2. [generator_3](/src/models/neg_cfg_generator.py#83)
3. [generator_2](/src/models/mood_generator.py#L151)
4. [generator_1](/src/models/generator.py#L140) *(deprecated)*
5. [minimal_generator_0](/src/models/minimal_generator.py#74) *(deprecated)*

### Available Generator Commands:
1. ***epoch:*** select a specific epoch number to use in inference  
└── ***opts:*** `0 < epoch < Config.EPOCHS (if not explicitly trained for more epochs)`
2. ***length:*** number of generated tokens in sequence  
└── ***opts:*** `0 < length < inf`
3. ***mood:*** select the target mood  
└── ***opts:*** `["angry", "exciting", "fear", "funny", "happy", "lazy", "magnificent", "quiet", "romantic", "sad", "warm",]`
4. ***transition-mood:*** select the mood to transition into  
└── ***opts:*** `["angry", "exciting", "fear", "funny", "happy", "lazy", "magnificent", "quiet", "romantic", "sad", "warm",]`
5. ***transition-step:*** one-based index of the step at which the mood shifts from first to second target (must be less than `--length`)  
└── ***opts:*** `0 < transition-step < length`
6. ***output:*** name of MIDI file where the tokens will be saved  
└── ***opts:*** `valid filename`

# Project Architecture
## a. Data setup

Data Directory:

```
.
└── datasets/
    ├── CASE_full/
    └── ...
```

Datasets:

- [Case](https://springernature.figshare.com/articles/dataset/CASE_Dataset-full/8869157?file=16260497)
- [XMIDI](https://drive.google.com/file/d/1qDkSH31x7jN8X-2RyzB9wuxGji4QxYyA/view)

## Model Blueprint

To initialize a transformer model, subclass the `GeneralModelHandler` and define a custom `train_step()` method for your specific model and loss (see `MinimalGeneratorHandler` in `src/models/minimal_generator.py`). The handler manages the training loop and automatically saves model checkpoints; you just call `handler.train(dataloader, epochs)` and provide your model, optimizer, and loss function.