Quickstart
==========

This quickstart demonstrates how to use the `CCFSRFG1` algorithm — a cooperative co-evolutionary
algorithm (CCEA) variant with random feature grouping — to perform feature selection on the
Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

In this example, you will:

- Load the dataset using the `DataLoader` utility.
- Configure the dataset and algorithm from `.toml` files.
- Run the optimization process.

Code example
------------

.. code-block:: python

    import toml
    import importlib.resources
    from pyccea.coevolution import CCFSRFG1
    from pyccea.utils.datasets import DataLoader

    # Load dataset parameters
    with importlib.resources.open_text("pyccea.parameters", "dataloader.toml") as toml_file:
        data_conf = toml.load(toml_file)

    # Initialize the DataLoader with the specified dataset and configuration
    data = DataLoader(dataset="wdbc", conf=data_conf)
    # Prepare the dataset for the algorithm (e.g., preprocessing, splitting)
    data.get_ready()

    # Load algorithm-specific parameters
    with importlib.resources.open_text("pyccea.parameters", "ccfsrfg.toml") as toml_file:
        ccea_conf = toml.load(toml_file)

    # Initialize the cooperative co-evolutionary algorithm
    ccea = CCFSRFG1(data=data, conf=ccea_conf, verbose=False)
    # Start the optimization process
    ccea.optimize()

Output
------

The best feature subset found is stored in the attribute ``best_context_vector``, a binary array where:

- ``1`` indicates a selected feature.
- ``0`` indicates an unselected feature.
