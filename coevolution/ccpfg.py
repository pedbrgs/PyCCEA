import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from projection.vip import VIP
from coevolution.ccea import CCEA
from projection.cipls import CIPLS
from fitness.penalty import SubsetSizePenalty
from evaluation.wrapper import WrapperEvaluation
from sklearn.model_selection import StratifiedKFold
from cooperation.best import SingleBestCollaboration
from sklearn.cross_decomposition import PLSRegression
from cooperation.random import SingleRandomCollaboration
from decomposition.ranking import RankingFeatureGrouping
from initialization.random import RandomBinaryInitialization
from optimizers.genetic_algorithm import BinaryGeneticAlgorithm


class CCPFG(CCEA):
    """Cooperative Co-Evolutionary Algorithm with Projection-based Feature Grouping (CCPFG).

    Attributes
    ----------
    subcomp_sizes: list
        Number of features in each subcomponent.
    feature_idxs: np.ndarray
        List of feature indexes.
    """

    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        # Number of components to keep after projection
        self.n_components = self.conf["decomposition"]["n_components"]
        # Method used to distribute features into subcomponents
        self.method = self.conf["decomposition"]["method"]
        # Perform K-fold cross-validation to compute variable importances
        kfold = StratifiedKFold(n_splits=self.conf["decomposition"]["kfolds"],
                                shuffle=True,
                                random_state=self.seed)
        vips = list()
        if self.data.n_features > 1000:
            high_dim = True
            logging.info("Projection with Covariance-free Incremental Partial Least Squares (CIPLS).")
        else:
            logging.info("Projection with Partial Least Squares (PLS).")
            high_dim = False
        for train_idx, _ in kfold.split(self.data.X_train, self.data.y_train):
            projection_model = (
                CIPLS(n_components=self.n_components, copy=True)
                if high_dim else
                PLSRegression(n_components=self.n_components, copy=True)
            )
            projection_model.fit(X=self.data.X_train[train_idx].copy(),
                                 Y=self.data.y_train[train_idx].copy())
            vip = VIP(model=projection_model)
            vip.compute()
            vips.append(vip.importances)
            del projection_model, vip
        # The importance of a variable will be the average value of its importances in the k-folds
        importances = pd.DataFrame(vips).mean(axis=0).values
        # Instantiate ranking feature grouping using variable importances as scores
        self.decomposer = RankingFeatureGrouping(n_subcomps=self.n_subcomps,
                                                 subcomp_sizes=self.subcomp_sizes,
                                                 scores=importances,
                                                 method=self.method,
                                                 ascending=False)

    def _init_collaborator(self):
        """Instantiate collaboration method."""
        self.best_collaborator = SingleBestCollaboration()
        self.random_collaborator = SingleRandomCollaboration(seed=self.seed)

    def _init_evaluator(self):
        """Instantiate evaluation method."""
        evaluator = WrapperEvaluation(task=self.conf["wrapper"]["task"],
                                      model_type=self.conf["wrapper"]["model_type"],
                                      eval_function=self.conf["evaluation"]["eval_function"],
                                      eval_mode=self.conf["evaluation"]["eval_mode"])
        self.fitness_function = SubsetSizePenalty(evaluator=evaluator,
                                                  weights=self.conf["evaluation"]["weights"])

    def _init_subpop_initializer(self):
        """Instantiate subpopulation initialization method."""
        self.initializer = RandomBinaryInitialization(data=self.data,
                                                      subcomp_sizes=self.subcomp_sizes,
                                                      subpop_sizes=self.subpop_sizes,
                                                      collaborator=self.random_collaborator,
                                                      fitness_function=self.fitness_function)

    def _init_optimizers(self):
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        self.optimizers = list()
        # Instantiate an optimizer for each subcomponent
        for i in range(self.n_subcomps):
            optimizer = BinaryGeneticAlgorithm(subpop_size=self.subpop_sizes[i],
                                               n_features=self.subcomp_sizes[i],
                                               conf=self.conf)
            self.optimizers.append(optimizer)

    def _problem_decomposition(self):
        """Decompose the problem into smaller subproblems."""
        # Decompose features in the training set
        self.data.S_train, self.feature_idxs = self.decomposer.decompose(X=self.data.X_train)
        # Update 'n_subcomps' when it starts with NoneType
        self.n_subcomps = self.decomposer.n_subcomps
        # Update 'subcomp_sizes' when it starts with an empty list
        self.subcomp_sizes = self.decomposer.subcomp_sizes.copy()
        # Reorder train data according to shuffling in feature decomposition
        self.data.X_train = self.data.X_train[:, self.feature_idxs].copy()

        # Train-validation
        if self.conf["evaluation"]["eval_mode"] == 'train_val':
            # Decompose features in the validation set
            self.data.S_val, _ = self.decomposer.decompose(X=self.data.X_val,
                                                           feature_idxs=self.feature_idxs)
            # Reorder validation data according to shuffling in feature decomposition
            self.data.X_val = self.data.X_val[:, self.feature_idxs].copy()
        # Cross-validation
        else:
            # It is just to avoid crashing when initializing the optimizers.
            # It will not be used in the cross-validation mode.
            self.data.S_val = np.full(shape=(self.n_subcomps), fill_value=None)

    def optimize(self):
        """Solve the feature selection problem through optimization."""
        # Decompose problem
        self._problem_decomposition()
        # Initialize subpopulations
        self._init_subpopulations()
        # Instantiate optimizers
        self._init_optimizers()

        # Get the best individual and context vector from each subpopulation
        self.current_best = self._get_best_individuals(
            subpops=self.subpops,
            fitness=self.fitness,
            context_vectors=self.context_vectors
        )
        # Select the globally best context vector
        self.best_context_vector, self.best_fitness = self._get_global_best()
        # Save the order of features considered in the random feature grouping
        self.best_feature_idxs = self.feature_idxs.copy()

        # Set the number of generations counter
        n_gen = 0
        # Number of generations that the best fitness has not improved
        stagnation_counter = 0
        # Initialize the optimization progress bar
        progress_bar = tqdm(total=self.conf["coevolution"]["max_gen"],
                            desc="Generations",
                            leave=False)

        # Iterate up to the maximum number of generations
        while n_gen <= self.conf["coevolution"]["max_gen"]:
            # Append current best fitness
            self.convergence_curve.append(self.best_fitness)
            # Evolve each subpopulation using a genetic algorithm
            for i in range(self.n_subcomps):
                self.subpops[i], self.fitness[i] = self.optimizers[i].evolve(
                    subpop=self.subpops[i],
                    fitness=self.fitness[i]
                )
            # For each subpopulation
            for i in range(self.n_subcomps):
                # Find best individuals from the previous generation as collaborators for each
                # individual in the current generation
                for j in range(self.subpop_sizes[i]):
                    collaborators = self.best_collaborator.get_collaborators(
                        subpop_idx=i,
                        indiv_idx=j,
                        subpops=self.subpops,
                        current_best=self.current_best
                    )
                    context_vector = self.best_collaborator.build_context_vector(collaborators)
                    # Update the context vector
                    # TODO Should I store the best context vector of each subpopulation across generations?
                    self.context_vectors[i][j] = context_vector.copy()
                    # Update fitness
                    self.fitness[i][j] = self.fitness_function.evaluate(context_vector, self.data)
            # Get the best individual and context vector from each subpopulation
            self.current_best = self._get_best_individuals(
                subpops=self.subpops,
                fitness=self.fitness,
                context_vectors=self.context_vectors
            )
            # Select the globally best context vector
            best_context_vector, best_fitness = self._get_global_best()
            # Update best context vector
            if self.best_fitness < best_fitness:
                # Reset stagnation counter because best fitness has improved
                stagnation_counter = 0
                # Enable logger if specified
                logging.getLogger().disabled = False if self.verbose else True
                # Objective weight
                w1 = self.conf["evaluation"]["weights"][0]
                # Penalty weight
                w2 = self.conf["evaluation"]["weights"][1]
                # Current fitness, performance evaluation and penalty
                current_best_fitness = round(self.best_fitness, 4)
                current_penalty = round(self.best_context_vector.sum()/self.n_features, 4)
                current_eval = round((self.best_fitness + w2*current_penalty)/w1, 4)
                # New fitness, performance evaluation and penalty
                new_best_fitness = round(best_fitness, 4)
                new_penalty = round(best_context_vector.sum()/self.n_features, 4)
                new_eval = round((best_fitness + w2*new_penalty)/w1, 4)
                # Show improvement
                logging.info(
                    f"\nUpdate fitness from {current_best_fitness} to {new_best_fitness}.\n"
                    f"Update predictive performance from {current_eval} to {new_eval}.\n"
                    f"Update penalty from {current_penalty} to {new_penalty}.\n"
                )
                # Update best context vector
                self.best_context_vector = best_context_vector.copy()
                # Update best fitness
                self.best_fitness = best_fitness
            else:
                # Increase stagnation counter because best fitness has not improved
                stagnation_counter += 1
                # Checks whether the optimization has been stagnant for a long time
                if stagnation_counter >= self.conf["coevolution"]["max_gen_without_improvement"]:
                    # Enable logger
                    logging.getLogger().disabled = False
                    logging.info(
                        "\nEarly stopping because fitness has been stagnant for "
                        f"{stagnation_counter} generations in a row."
                    )
                    break
            # Increase number of generations
            n_gen += 1
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar after optimization
        progress_bar.close()
