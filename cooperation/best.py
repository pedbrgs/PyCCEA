import numpy as np
from cooperation.collaboration import Collaboration


class SingleBestCollaboration(Collaboration):
    """
    Set the best individual from each subpopulation as a collaborator to any individual.
    """

    def get_best_individuals(self,
                             subpops: list,
                             local_fitness: list,
                             global_fitness: list,
                             context_vectors: list):
        """
        Get the best individual from each subpopulation.

        Parameters
        ----------
        subpops: list
            Individuals from all subpopulations. Each individual is represented by a binary
            n-dimensional array, where n is the number of features. If there is a 1 in the i-th
            position of the array, it indicates that the i-th feature should be considered and if
            there is a 0, it indicates that the feature should not be considered.
        local_fitness: list
            Evaluation of all individuals from all subpopulations.
        global_fitness: list
            Evaluation of all context vectors from all subpopulations.
        context_vectors: list
            Complete problem solutions.

        Returns
        -------
        current_best: dict
            Current best individual of each subpopulation and its respective evaluation.
        """
        # Current best individual of each subpopulation
        current_best = dict()
        # Number of subpopulations
        n_subpops = len(subpops)
        # For each subpopulation
        for i in range(n_subpops):
            # TODO Would the best individual be the one with the highest global fitness or the highest local fitness?
            # best_ind_idx = np.argmax(self.local_fitness[i])
            best_ind_idx = np.argmax(global_fitness[i])
            current_best[i] = dict()
            current_best[i]["individual"] = subpops[i][best_ind_idx]
            current_best[i]["local_fitness"] = local_fitness[i][best_ind_idx]
            current_best[i]["context_vector"] = context_vectors[i][best_ind_idx]
            current_best[i]["global_fitness"] = global_fitness[i][best_ind_idx]

        return current_best

    def get_collaborators(self,
                          subpop_idx: int,
                          indiv_idx: int,
                          subpops: list,
                          current_best: dict):
        """
        Set the best individual from each subpopulation as a collaborator to the individual given
        as a parameter.

        Parameters
        ----------
        subpop_idx: int
            Index of the subpopulation to which the individual belongs.
        indiv_idx: int
            Index of the individual in its respective subpopulation.
        subpops: list
            Individuals from all subpopulations.
        current_best: dict
            Current best individual of each subpopulation and its respective evaluation.

        Returns
        -------
        collaborators: list
            A single individual from each subpopulation that will collaborate with the individual
            given as a parameter.
        """
        # Best individuals from each subpopulation as collaborators
        collaborators = [best["individual"] for best in current_best.values()]
        # Assign the individual itself to the subpopulation to which it belongs
        collaborators[subpop_idx] = subpops[subpop_idx][indiv_idx]

        return collaborators
