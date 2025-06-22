Tutorials
=========

This section provides step-by-step tutorials on how to use and customize cooperative
co-evolutionary algorithms for feature selection using the PyCCEA package.

.. contents::
   :local:
   :depth: 2
   :backlinks: entry

How to use a baseline CCEA?
---------------------------

This tutorial demonstrates how to use the ``CCFSRFG1`` algorithm — a cooperative co-evolutionary
algorithm (CCEA) variant with random feature grouping — to perform feature selection on the
Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

In this example, you will:

- Load the dataset using the ``DataLoader``.
- Load the dataset and algorithm configuration files.
- Run the optimization process.

We start by importing the necessary modules and classes:

.. code-block:: python

    import toml
    import importlib.resources
    from pyccea.coevolution import CCFSRFG1
    from pyccea.utils.datasets import DataLoader

- ``toml`` is used to parse configuration files.
- ``importlib.resources`` helps access files inside a package.
- ``CCFSRFG1`` is the cooperative co-evolution algorithm you'll run.
- ``DataLoader`` is a utility to prepare the dataset.

Next, we load the configuration for the dataset from a `.toml` file:

.. code-block:: python

    with importlib.resources.open_text("pyccea.parameters", "dataloader.toml") as toml_file:
        data_conf = toml.load(toml_file)

This code looks for the file ``dataloader.toml`` inside the ``pyccea.parameters`` package and loads
its content into a dictionary.

We then create and prepare the dataset using the ``DataLoader``:

.. code-block:: python

    dataloader = DataLoader(dataset="wdbc", conf=data_conf)
    dataloader.get_ready()

- ``dataset="wdbc"`` specifies the dataset to use (WDBC in this case).
- ``conf=data_conf`` passes the parameters we just loaded.
- ``get_ready()`` performs any necessary preprocessing and splits the data.

We now load the configuration for the CCFSRFG1 algorithm:

.. code-block:: python

    with importlib.resources.open_text("pyccea.parameters", "ccfsrfg.toml") as toml_file:
        ccea_conf = toml.load(toml_file)

This step reads the ``ccfsrfg.toml`` file and loads the algorithm's hyperparameters into a
dictionary.

Now we're ready to run the algorithm:

.. code-block:: python

    ccea = CCFSRFG1(data=dataloader, conf=ccea_conf, verbose=False)
    ccea.optimize()

- ``CCFSRFG1(...)`` initializes the algorithm with data and configuration.
- ``optimize()`` starts the evolutionary process for feature selection.

Once the optimization is complete, the best feature subset is stored in the ``best_context_vector``
attribute of the ``CCFSRFG1`` object, a binary vector where:

  - `1` indicates that a feature is selected.
  - `0` indicates that a feature is excluded.

Some CCEAs, like ``CCFSRFG1``, may reorder the features during the decomposition phase. In these
cases, you must map the selected features back to their original names using the reordered
indices.

.. code-block:: python

    # Get original feature column names
    feature_cols = dataloader.data.columns
    # Reorder the kept features according to the algorithm's internal feature order
    reordered_features = feature_cols[ccea.feature_idxs]
    # Select features where best_context_vector == 1
    selected_features = reordered_features[ccea.best_context_vector.astype(bool)].tolist()

If you experiment with other CCEAs, such as ``CCPSTFG``, be aware that some features may be
removed before optimization. For those, you might need to filter out removed features before
reordering:

.. code-block:: python

    # Create a set of indices for features that were not removed
    kept_feature_indices = set(range(len(feature_cols))) - ccea.removed_features
    # Select the original feature names corresponding to the kept indices
    kept_feature_names = feature_cols[list(kept_feature_indices)]
    # Reorder the kept features according to the algorithm's internal feature order
    reordered_features = kept_feature_names[ccea.feature_idxs]
    # Select features where best_context_vector == 1
    selected_features = reordered_features[ccea.best_context_vector.astype(bool)].tolist()

Feel free to modify the configuration files or try different CCEAs to explore various scenarios.


How to customize a CCEA?
------------------------

You can implement your own cooperative co-evolutionary algorithm in PyCCEA by subclassing the
:py:class:`pyccea.coevolution.ccea.CCEA` base class and customizing its core components.

To do so, your new class should implement or override the following methods:

- ``_init_collaborator()``: defines how individuals from different subcomponents collaborate.
- ``_init_evaluator()``: specifies how candidate solutions are evaluated.
- ``_init_subpop_initializer()``: sets up how initial solutions are generated.
- ``_init_optimizers()``: chooses the evolutionary algorithm used for optimization.
- ``_init_decomposer()``: defines the decomposition strategy (feature grouping).
- ``optimize()``: runs the main optimization loop.

We will use `CCEAFS` (Cooperative Co-Evolutionary-Based Feature Selection) as a concrete example
to illustrate the necessary steps.

1. **_init_collaborator**

This method instantiates collaboration strategies that define how individuals from different
subpopulations interact to form a full candidate solution (context vector) during evaluation.
For example:

.. code-block:: python

    def _init_collaborator(self):
        """Instantiate collaboration method."""
        self.best_collaborator = SingleBestCollaboration()
        self.random_collaborator = SingleRandomCollaboration(seed=self.seed)

- ``SingleBestCollaboration()`` typically selects the best individual from each cooperating
  subpopulation, used when evaluating evolved individuals.
- ``SingleRandomCollaboration(seed=self.seed)`` selects random collaborators, useful for
  initialization to encourage diversity.

2. **_init_evaluator**

This method sets up the fitness function that evaluates complete candidate solutions and guides
the evolutionary process. For example:

.. code-block:: python

    def _init_evaluator(self):
        """Instantiate evaluation method."""
        evaluator = WrapperEvaluation(
            task=self.conf["wrapper"]["task"],
            model_type=self.conf["wrapper"]["model_type"],
            eval_function=self.conf["evaluation"]["eval_function"],
            eval_mode=self.eval_mode,
            n_classes=getattr(self.data, "n_classes", None)
        )
        self.fitness_function = SubsetSizePenalty(
            evaluator=evaluator,
            weights=self.conf["evaluation"]["weights"]
        )

Here, ``WrapperEvaluation`` typically wraps a model evaluation, while ``SubsetSizePenalty`` adds a
penalty term to encourage compact feature subsets.

3. **_init_subpop_initializer**

This method defines how to generate initial subpopulations of individuals, marking the start of
co-evolution. For example:

.. code-block:: python

    def _init_subpop_initializer(self):
        """Instantiate subpopulation initialization method."""
        self.initializer = RandomBinaryInitialization(
            data=self.data,
            subcomp_sizes=self.subcomp_sizes,
            subpop_sizes=self.subpop_sizes,
            collaborator=self.random_collaborator,
            fitness_function=self.fitness_function
        )

``RandomBinaryInitialization`` suggests each individual is a binary vector (e.g., feature
inclusion/exclusion) initialized randomly.

4. **_init_optimizers**

This method instantiates evolutionary optimizers to independently evolve each subpopulation.
For example:

.. code-block:: python

    def _init_optimizers(self):
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        self.optimizers = []
        for i in range(self.n_subcomps):
            optimizer = BinaryGeneticAlgorithm(
                subpop_size=self.subpop_sizes[i],
                n_features=self.subcomp_sizes[i],
                conf=self.conf
            )
            self.optimizers.append(optimizer)

Each subpopulation uses a ``BinaryGeneticAlgorithm`` tailored for binary representations.

5. **_init_decomposer**

This method defines how the original problem is split into subproblems (subcomponents),
each handled by a separate subpopulation. For example:

.. code-block:: python

    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        self.decomposer = SequentialFeatureGrouping(
            n_subcomps=self.n_subcomps,
            subcomp_sizes=self.subcomp_sizes
        )

``SequentialFeatureGrouping`` divides features into sequential groups; decomposition strategies
vary and strongly affect algorithm performance.

6. **optimize**

This is the main method orchestrating the entire co-evolutionary optimization workflow,
including decomposition, initialization, evolution, evaluation, and convergence checks.

.. code-block:: python

    def optimize(self):
        """Solve the feature selection problem through optimization."""
        self._problem_decomposition()
        self._init_subpopulations()
        self._init_optimizers()

        self.current_best = self._get_best_individuals(
            subpops=self.subpops,
            fitness=self.fitness,
            context_vectors=self.context_vectors
        )
        self.best_context_vector, self.best_fitness = self._get_global_best()
        self.best_context_vectors.append(self.best_context_vector.copy())
        self.best_feature_idxs = self.feature_idxs.copy()

        n_gen = 0
        stagnation_counter = 0

        while n_gen <= self.conf["coevolution"]["max_gen"]:
            self.convergence_curve.append(self.best_fitness)

            # Evolve subpopulations independently
            current_subpops = [
                self.optimizers[i].evolve(self.subpops[i], self.fitness[i])
                for i in range(self.n_subcomps)
            ]

            # Evaluate evolved individuals collaboratively
            current_fitness, current_context_vectors = [], []
            for i in range(self.n_subcomps):
                current_fitness.append([])
                current_context_vectors.append([])
                for j in range(self.subpop_sizes[i]):
                    collaborators = self.best_collaborator.get_collaborators(
                        subpop_idx=i,
                        indiv_idx=j,
                        current_subpops=current_subpops,
                        current_best=self.current_best
                    )
                    context_vector = self.best_collaborator.build_context_vector(collaborators)
                    current_context_vectors[i].append(context_vector.copy())
                    fitness_value = self.fitness_function.evaluate(context_vector, self.data)
                    current_fitness[i].append(fitness_value)

            # Update subpopulations and fitness
            self.subpops = copy.deepcopy(current_subpops)
            self.fitness = copy.deepcopy(current_fitness)
            self.context_vectors = copy.deepcopy(current_context_vectors)

            self.current_best = self._get_best_individuals(
                subpops=self.subpops,
                fitness=self.fitness,
                context_vectors=self.context_vectors
            )

            best_context_vector, best_fitness = self._get_global_best()

            if self.best_fitness < best_fitness:
                stagnation_counter = 0
                self.best_context_vector = best_context_vector.copy()
                self.best_context_vectors.append(self.best_context_vector.copy())
                self.best_fitness = best_fitness
            else:
                stagnation_counter += 1
                if stagnation_counter >= self.conf["coevolution"]["max_gen_without_improvement"]:
                    break

            n_gen += 1

The ``optimize`` method demonstrates the typical CCEA workflow:

- **Decomposition and initialization:** Decompose the problem and initialize subpopulations and
  optimizers.
- **Evolution loop:** Repeat for a maximum number of generations or until stagnation.
- **Subpopulation evolution:** Evolve each subpopulation independently.
- **Collaboration and evaluation:** Collaborate across subpopulations to build complete solutions
  and evaluate fitness.
- **Best solution tracking:** Update global best solution and check convergence.

By overriding these key ``_init_*`` methods and ``optimize()``, you can flexibly customize the
co-evolutionary process to fit your problem domain, defining collaboration, evaluation,
initialization, optimization, decomposition, and the optimization flow itself.
