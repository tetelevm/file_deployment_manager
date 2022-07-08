import random
from math import log

try:
    from .algorithm_adapter import BaseAlgorithm, DeploymentMatrix
except ImportError:
    # if run as "__main__"
    from algorithm_adapter import BaseAlgorithm, DeploymentMatrix, abstract_main


__all__ = [
    "AntColony",
]


class Ant:
    """
    A unit for collecting statistics.
    Each time it collects, it recreates its deployment matrix depending
    on the current probability matrix.
    """

    def __init__(self, colony):
        colony: AntColony
        self.matrix = colony.matrix.copy()
        self.weights = colony.matrix_weights  # link, not copy

    @staticmethod
    def get_choose(probability: float) -> int:
        # probability is not zero
        return random.choices((0, 1), weights=(1/probability, probability))[0]

    def make_matrix(self) -> DeploymentMatrix:
        """
        The main unit method, randomly selects a value for each cell
        depending on the weight.
        """
        for f_ind in range(self.matrix.f_count):
            for sv_ind in range(self.matrix.sv_count):
                self.matrix[f_ind, sv_ind] = self.get_choose(self.weights[f_ind][sv_ind])
        return self.matrix


class AntColony(BaseAlgorithm):
    """
    A class that realizes the ant colony algorithm in the context of the
    project.
    The idea of the algorithm is that it collects its statistics for the
    best variants. The statistics are collected randomly using
    individual units (ants), and are gradually changed depending on the
    new statistics, improving the final result.
    The statistics (pheromones) are collected in a matrix of size
    `[files x servers]`. Each cell is a float with a probability value
    of whether a particular file should be stored on a particular
    server. Gradually the probability decreases, but increases for good
    choices, thus showing the best choices and ignoring the bad ones.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        all_count = self.matrix.f_count * self.matrix.sv_count
        self.scout_number = 0
        self.scout_count = all_count

        self.ant_count = all_count * 4
        self.evaporation_coefficient = min(0.7, 1/log(log(all_count)))
        self.threshold = 1/log(all_count, 2)

        self.matrix_weights: list[list[float]] = [
            [1] * self.matrix.sv_count
            for _ in range(self.matrix.f_count)
        ]
        self.ants = [Ant(self) for _ in range(self.ant_count)]

    def vaporize_pheromones(self):
        """
        A method that decreases all weights. Uses a threshold below
        which the weight cannot fall.
        """

        for file_i in range(self.matrix.f_count):
            for server_i in range(self.matrix.sv_count):
                if self.matrix_weights[file_i][server_i] > self.threshold:
                    self.matrix_weights[file_i][server_i] *= self.evaporation_coefficient
                if self.matrix_weights[file_i][server_i] < self.threshold:
                    self.matrix_weights[file_i][server_i] = self.threshold

    def spray_pheromones(self, matrix: DeploymentMatrix, weight: float):
        """
        A method that increases the weights from the matrix cells used
        in the current version by a given weight.
        """

        for file_i in range(matrix.f_count):
            for server_i in range(matrix.sv_count):
                if matrix[file_i, server_i]:
                    self.matrix_weights[file_i][server_i] += weight

    def update_pheromones(self, variants: list[DeploymentMatrix]):
        """
        A method for updating pheromones.
        First calculates all weights needed for updating (the better
        result, the bigger weight), reduces all pheromones and sets new
        ones for all variants.
        """

        best_result = self.best_value
        deltas = [
            self.get_deployment_result(matrix) - best_result
            for matrix in variants
        ]
        if all(delta == 0 for delta in deltas):
            # only the current result or all the same
            weights = [1] * len(deltas)
        else:
            worst_delta = deltas[-1]
            weights = [(1 - (delta / worst_delta)) ** 4 for delta in deltas]

        self.vaporize_pheromones()
        for (matrix, weight) in zip(variants, weights):
            self.spray_pheromones(matrix, weight)

    def filter_variants(self, variants: list[DeploymentMatrix]) -> list[DeploymentMatrix]:
        """
        Filters and sorts variants.
        Variants are taken only possible for deployment, not less
        optimal than the current variant, and only unique (the same
        variants are taken only once).
        """

        unique_variants = [self.matrix]
        for matrix in variants:
            if not self.check_prerequisite(matrix):
                continue
            if self.get_deployment_result(matrix) > self.best_value:
                continue
            for current_matrix in unique_variants:
                if matrix == current_matrix:
                    break
            else:
                unique_variants.append(matrix)

        sorted_paths = sorted(unique_variants, key=self.get_deployment_result)
        return sorted_paths

    def explore_variants(self):
        """
        A method that explores variants and updates its statistics.
        First it creates variants using a colony of units, then filters
        them, and then updates the weights and its best score based on
        the variants.
        """

        variants = [ant.make_matrix() for ant in self.ants]
        sorted_variants = self.filter_variants(variants)

        if self.best_value > self.get_deployment_result(sorted_variants[0]):
            self.matrix = sorted_variants[0].copy()
            self.best_value = self.get_deployment_result(self.matrix)

        self.update_pheromones(sorted_variants)

    def do_one_step(self):
        """
        Takes a step to update the weights and prints the result.
        Updating goes up to a certain (rather abstract) point, during
        which time the optimal variant is calculated.
        """

        self.scout_number += 1
        self.explore_variants()
        print(f"{self.scout_number: <8}  ==  {self.best_value}")
        if self.scout_number > self.scout_count:
            self.stop = True


if __name__ == "__main__":
    abstract_main(AntColony)
