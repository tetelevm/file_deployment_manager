import random
from math import log

try:
    from .algorithm_adapter import BaseAlgorithm, DeploymentMatrix
except ImportError:
    # if run as "__main__"
    from algorithm_adapter import BaseAlgorithm, DeploymentMatrix, abstract_main

__all__ = [
    "GeneticAlgorithm",
]

flip_coin = lambda: random.random() > 0.5
POPULATION_TYPE = list[DeploymentMatrix]


class GeneticAlgorithm(BaseAlgorithm):
    """
    A class that realizes the genetic algorithm in the context of the
    project.
    The idea behind the algorithm is that each iteration creates a new
    generation of elements, slightly different from their parents. And
    in each generation, only a part of the best versions survive, which
    interbreed with each other and create a new generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        all_count = self.matrix.f_count * self.matrix.sv_count
        self.population_number = 0
        self.population_number_max = 100 * 1.1 ** log(all_count, 1.6)
        self.child_count = 10
        self.count_best = 10
        self.population = self.create_descendants(self.matrix, self.count_best)

    @staticmethod
    def create_descendants(parent_matrix: DeploymentMatrix, count: int) -> POPULATION_TYPE:
        """
        Copies the matrix the specified count of times.
        """
        return [parent_matrix.copy() for _ in range(count)]

    def change_existence(self, matrix: DeploymentMatrix):
        """
        Randomly changes the location of one file on one server.
        """

        f_ind = random.randrange(matrix.f_count)
        s_ind = random.randrange(matrix.sv_count)
        matrix[f_ind, s_ind] ^= 1

    def swap_existence(self, matrix: DeploymentMatrix):
        """
        Randomly moves one file to another server.
        """

        f = random.randrange(matrix.f_count)
        s_1, s_2 = random.sample(range(matrix.sv_count), k=2)
        matrix[f, s_1], matrix[f, s_2] = matrix[f, s_2], matrix[f, s_1]

    def mutate_population(self, population: POPULATION_TYPE) -> POPULATION_TYPE:
        """
        Mutates everyone within the population and chooses the best
        among the resulting ones.
        The resulting population will be at least `.count_best' and all
        viable, even if all the resulting matrices are unviable (in this
        case the current solution is substituted for the bad matrices).
        Modifies the resulting object.
        """

        for matrix in population:
            if flip_coin():
                self.change_existence(matrix)
            if flip_coin():
                self.swap_existence(matrix)

        population = list(filter(self.check_prerequisite, population))
        missing_count = self.count_best - len(population)
        if missing_count > 0:
            missing = self.create_descendants(self.matrix, missing_count)
            population.extend(missing)

        return population

    def crossbreed_matrix(
            self,
            matrix_1: DeploymentMatrix,
            matrix_2: DeploymentMatrix,
    ) -> DeploymentMatrix:
        """
        It interbreed the two parents.
        In those cells where the parents have different values, a cell
        of one of the parents is randomly chosen.
        If the descendant is unviable, one of the parents returns.
        """

        child = matrix_1.copy()
        for f_ind in range(matrix_1.f_count):
            for sv_ind in range(matrix_1.sv_count):
                if matrix_1[f_ind, sv_ind] != matrix_2[f_ind, sv_ind]:
                    if flip_coin():
                        child[f_ind, sv_ind] = matrix_2[f_ind, sv_ind]

        if self.check_prerequisite(child):
            return child
        else:
            return matrix_1 if flip_coin else matrix_2

    def crossbreed_population(self, population: POPULATION_TYPE) -> POPULATION_TYPE:
        """
        It interbreeds a population with each other.
        For each parent there are necessarily at least `.child_count`
        descendants. Increases the population `.child_count` times.
        """

        weights = [1] * self.count_best
        hybrids = []
        for current_ind in range(self.count_best):
            first_parent = population[current_ind]
            weights[current_ind] = 0
            other_parents = random.choices(population, weights, k=self.child_count)
            weights[current_ind] = 1
            for second_parent in other_parents:
                hybrids.append(self.crossbreed_matrix(first_parent, second_parent))

        return hybrids

    def grow_generation(self):
        """
        Grows the next generation.
        First mutates the current generation, then selects the most
        successful results from it and crosses them with each other.
        """

        new_generation = self.mutate_population(self.population)
        new_generation += [self.matrix]  # if all solutions are worse than the original

        best_from_generation = sorted(new_generation, key=self.get_deployment_result)
        best_from_generation = best_from_generation[:self.count_best]

        self.matrix = best_from_generation[0].copy()
        self.best_value = self.get_deployment_result()

        hybrids = self.crossbreed_population(best_from_generation)
        self.population = hybrids + [self.matrix]

        self.matrix = min(self.population, key=self.get_deployment_result).copy()
        self.best_value = self.get_deployment_result()

    def do_one_step(self):
        """
        Grows a generation and prints the results. The stopping point is
        a fairly late generation, which is very likely optimal.
        """

        self.population_number += 1
        self.grow_generation()

        if self.print_logs:
            print(f"{self.population_number: <8}  ==  {self.best_value}")
        if self.population_number >= self.population_number_max:
            self.stop = True


if __name__ == "__main__":
    abstract_main(GeneticAlgorithm)
