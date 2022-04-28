"""
Project master file. Connects all the other parts together, implementing
this algorithm:

- parse configs
- parse of input data
- time matrix calculation
- calculation of the result using the desired algorithms
"""


from config_parser import ConfigParser
from matrix_calculator import MatrixCalculator, TimeMatrix
from algorithms import (
    AntColony,
    BeesColony,
    BranchAndBound,
    GeneticAlgorithm,
    SimulatedAnnealing,
)


available_algorithms = {
    "ant_colony": AntColony,
    "bees_colony": BeesColony,
    "branch_and_bound": BranchAndBound,
    "genetic_algorithm": GeneticAlgorithm,
    "simulated_annealing": SimulatedAnnealing,
}


def main():
    pass


if __name__ == "__main__":
    main()
