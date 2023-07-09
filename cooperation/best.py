from cooperation.collaboration import Collaboration


class SingleBestCollaboration(Collaboration):
    """
    Set the best individual from each subpopulation as a collaborator to any individual.
    """

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
