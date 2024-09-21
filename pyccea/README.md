# Quick start

## Introduction

The PyCCEA package is designed to simplify the use of cooperative co-evolutionary algorithms (CCEA) for feature selection tasks. This quick start guide will walk you through loading a dataset, configuring parameters, and running the optimization process using the CCFSRFG1 algorithm, a cooperative co-evolutionary algorithm with random feature grouping.

In this tutorial, we will:

- Load the Wisconsin Diagnostic Breast Cancer (WDBC) dataset using the `DataLoader` utility.
- Configure the algorithm and dataset settings from `.toml` files.
- Initialize and run the CCFSRFG1 algorithm to optimize the feature selection problem.

The dataset and algorithm parameters are defined in separate `.toml` configuration files, making it easy to manage and adjust settings without modifying the code itself.

## Example

By following this example, you will see how easy it is to use PyCCEA to solve feature selection problems with just a few lines of code. Let’s walk through the process of loading data, configuring the algorithm, and running the optimization:

```python
import toml
import importlib.resources
from pyccea.coevolution import CCFSRFG1
from pyccea.utils.datasets import DataLoader

# Load dataset parameters
with importlib.resources.open_text("pyccea.parameters", "dataloader.toml") as toml_file:
    data_conf = toml.load(toml_file)

# Initialize the DataLoader with the specified dataset and configuration
data = DataLoader(
    dataset="wdbc",
    conf=data_conf
)

# Prepare the dataset for the algorithm (e.g., preprocessing, splitting)
data.get_ready()

# Load algorithm-specific parameters
with importlib.resources.open_text("pyccea.parameters", "ccfsrfg.toml") as toml_file:
    ccea_conf = toml.load(toml_file)

# Initialize the cooperative co-evolutionary algorithm
ccea = CCFSRFG1(data=data, conf=ccea_conf, verbose=False)

# Start the optimization process
ccea.optimize()
```

The best solution can be found in the attribute `ccea.best_context_vector`, which is a binary vector where 1's indicate selected features and 0's represent non-selected features.