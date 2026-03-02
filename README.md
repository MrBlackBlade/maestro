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

# Project Architecture
## Data setup

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