# **EISG**

**TU benchmark**：D&D, MUTAG, PROTEINS.

 **DrugOOD**：IC50/EC50-size/scaffold/assay.

### Folder Specification

- `data/`: containing the data for training, including the preprocess result and substructure information merging scripts.
- `config/`: configuration for model and model training.
- `teain_eval.py`: the script to train our algorithm.
- `figure/`: containing the visualization of QED dataset.
- `models/`: containing backbone definition for our method.
- `losses.py`: containing the loss definition.
- `main.py`: the script to run.
- `vis.py`: the script to visualization.

### Package Dependency

```
python: 3.9.19
pytorch: 1.12.1          # With CUDA 11.3 support
torch-geometric: 2.6.0    # PyTorch Geometric (PyG)
torchvision: 0.13.1       # Computer vision extension
torchaudio: 0.12.1        # Audio processing extension
drugood: 0.0.1            # Drug discovery toolkit
rdkit: 2023.9.5           # Cheminformatics toolkit (latest stable)
numpy: 1.26.4             # Numerical computing foundation
cudatoolkit: 11.3.1       # CUDA toolkit
dgl-cu110: 0.6.1          # Deep Graph Library (CUDA 11 compatible)
datasets: 2.20.0          # HuggingFace dataset handling
transformers: 4.43.4      # Pretrained models library
scikit-learn: 1.2.2       # Machine learning toolkit
pandas: 2.2.2             # Data analysis framework
```

###  Data Generation

- **DruOOD** ：The first step is to generate the original dataset from CHEMBL database. As for the detailed process or operation, please refer to the [DrugOOD](https://github.com/tencent-ailab/DrugOOD) repository. The generated files should be put into folder or respectively.
- `json``DrugOOD/data/ic50``DrugOOD/data/ec50`
- **TU** ：auto download.

### Run the Code

run main.py
