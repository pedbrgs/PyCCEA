<p align="center">
    <img width="150" src="./docs/figures/logo.png" alt="PyCCEA logo">
<p>

[![PyPI](https://img.shields.io/pypi/v/pyccea.svg)](https://pypi.org/project/pyccea/)
[![codecov](https://codecov.io/gh/pedbrgs/PyCCEA/branch/main/graph/badge.svg?token=4X58GY16J4)](https://codecov.io/gh/pedbrgs/PyCCEA)
[![status](https://joss.theoj.org/papers/dd1af373a617c14953d289c085c38d8f/status.svg)](https://joss.theoj.org/papers/dd1af373a617c14953d289c085c38d8f)
![License](https://img.shields.io/pypi/l/pyccea)
![Python Versions](https://img.shields.io/pypi/pyversions/pyccea)
[![Downloads](https://pepy.tech/badge/pyccea)](https://pepy.tech/project/pyccea)



***

## :bulb: Overview

PyCCEA is an open-source package developed as part of ongoing doctoral research. It provides cooperative co-evolutionary strategies tailored for feature selection in large-scale and high-dimensional problems. The framework adopts a modular, decomposition-based approach and is intended for researchers and practitioners tackling complex feature selection tasks.

> **Note:** PyCCEA is a work in progress. Stay tuned for improvements and new algorithm implementations.

## :computer: Installation

To install the PyCCEA package directly from PyPI, use the following command in a Python ≥ 3.10 environment:

```
pip install pyccea
```

Alternatively, if you want to install the latest version directly from the GitHub:


```
pip install git+https://github.com/pedbrgs/pyccea.git
```

Ensure you have `pip` and an active internet connection to download dependencies.

## :high_brightness: Quickstart

This quickstart demonstrates how to use the CCFSRFG1 algorithm — a CCEA variant with random feature grouping — to perform feature selection on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

In this example, you will:

- Load the dataset using the `DataLoader` utility.
- Configure the dataset and algorithm from `.toml` files.
- Run the optimization process.


```python
import toml
import importlib.resources
from pyccea.coevolution import CCFSRFG1
from pyccea.utils.datasets import DataLoader

# Load dataset parameters
with importlib.resources.open_text("pyccea.parameters", "dataloader.toml") as toml_file:
    data_conf = toml.load(toml_file)

# Initialize the DataLoader with the specified dataset and configuration
dataloader = DataLoader(dataset="wdbc", conf=data_conf)
# Prepare the dataset for the algorithm (e.g., preprocessing, splitting)
dataloader.get_ready()

# Load algorithm-specific parameters
with importlib.resources.open_text("pyccea.parameters", "ccfsrfg.toml") as toml_file:
    ccea_conf = toml.load(toml_file)

# Initialize the cooperative co-evolutionary algorithm
ccea = CCFSRFG1(data=dataloader, conf=ccea_conf, verbose=False)
# Start the optimization process
ccea.optimize()
```

The best feature subset found is stored in the attribute `best_context_vector`, a binary array where 1 indicates a selected feature and 0 indicates an unselected one.

## :file_folder: Custom datasets

Custom datasets are supported as long as they conform to the PyCCEA input schema (`.parquet` file with feature columns and a `label` column). To register a custom dataset at runtime, add an entry to `DataLoader` and execute the standard preprocessing, splitting, and normalization pipeline:

```python
from pyccea.utils.datasets import DataLoader

# Path to your dataset in PyCCEA schema
data_path = "./custom_data.parquet"
dataset_name = "custom_data"

# Register the dataset path and task
DataLoader.DATASETS = {
    "task": "classification"  # or regression
    "file": data_path
}

# Load and prepare the dataset
dataloader = DataLoader(
    dataset_name=dataset_name,
    conf=data_conf
)
dataloader.get_ready()
```

If you prefer ready-to-use data, additional datasets already normalized to the PyCCEA format are available in the [High-Dimensional datasets repository](https://github.com/pedbrgs/High-Dimensional-Datasets/).

## :books: Documentation

Full documentation, including a comprehensive user guide, step-by-step tutorials, an API reference, and contribution guidelines, is available at [**PyCCEA docs**](https://pedbrgs.github.io/PyCCEA/).

## :scroll: Citation info

If you are using these codes in any way, please cite the following paper:

```
@article{PyCCEA,
    title = {PyCCEA: A Python package of cooperative co-evolutionary algorithms for feature selection in high-dimensional data},
    author = {Venancio, Pedro Vinicius A. B. and Batista, Lucas S.},
    journal = {Journal of Open Source Software},
    volume = {10},
    number = {112},
    pages = {8348},
    year = {2025}
}
```

***

## :mailbox: Contact
Please send any bug reports, questions or suggestions directly in the repository.
