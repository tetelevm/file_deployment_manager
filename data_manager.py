import json
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Type

from algorithms._algorithm_adapter import BaseAlgorithm
from algorithms import (
    AntColony,
    BeesColony,
    GeneticAlgorithm,
    SimulatedAnnealing,
)


__all__ = [
    "DataManager",
]


class DataManager:
    """
    A class that parses the algorithms to compute file locations (see
    `.get_algorithms()`) and reads the input data (see `.read_data()`).
    The data for the calculation is described in the file `data.json`.
    """

    available_algorithms: dict[str, Type[BaseAlgorithm]] = {
        "ant_colony": AntColony,
        "bees_colony": BeesColony,
        "genetic_algorithm": GeneticAlgorithm,
        "simulated_annealing": SimulatedAnnealing,
    }
    data_json_path = "./data.json"
    result_json_path = "./result.json"

    @classmethod
    def get_algorithms(cls) -> list[str]:
        """
        Parses program run flags. Returns a list of algorithm names.

        The start flags are the names of the algorithms you want to use
        to make calculations. The available names are taken from the
        `.available_algorithms` attribute and passed with one dash:
        --------------
        python3 ./main.py -bees_colony -ant_colony
        ['ant_colony', 'bees_colony']
        --------------
        python3 ./main.py -unknown_algorithm
        error: unrecognized arguments: -unknown_algorithm
        """
        parser = ArgumentParser()
        for algorithm_name in cls.available_algorithms:
            parser.add_argument("-" + algorithm_name, action=BooleanOptionalAction)
        parser = parser.parse_args()
        alg_names = [name for (name, val) in parser.__dict__.items() if val]
        if not alg_names:
            raise ValueError("Enter the optimization algorithms")
        return alg_names

    @classmethod
    def read_data(cls, path: str = None):
        """
        Reads the data from the `data.json` file  (placed in the root of
        the project or passed in an argument), and retrieves the
        calculation parameters from it.
        Ideally, json is not the best choice for this data format, but
        it is the only way to provide this data in a simple and readable
        way.
        """

        path = path or cls.data_json_path
        with open(path) as file:
            data: dict = json.load(file)

        coefficient = data.get("coefficient", 1)

        params = ["file_sizes", "delays", "pc_to_ls", "ls_to_cs", "cs_to_sv",
                  "server_prices", "server_spaces"]
        for name in params:
            param = data.get(name, None)
            if not param:
                raise ValueError(f"No <{name}> parameter")

        counts = data.get("counts", None)
        if counts is None:
            counts = {
                "files": len(data["file_sizes"]),
                "pc": len(data["pc_to_ls"]),
                "ls": len(data["ls_to_cs"]),
                "cs": len(data["cs_to_sv"]),
                "sv": len(data["cs_to_sv"][0]),
            }

        return (
            counts,
            data["file_sizes"],
            data["delays"],
            data["pc_to_ls"],
            data["ls_to_cs"],
            data["cs_to_sv"],
            data["server_prices"],
            data["server_spaces"],
            coefficient,
        )

    @classmethod
    def write_result(cls, result: dict, path: str = None):
        """
        Writes the results to a file.
        The results are written in json format to the specified path or
        to the default file (`result.json`).

        Record format:
        {
            algorithm_name: result_matrix,
            algorithm_name: result_matrix,
        }
        """

        path = path or cls.result_json_path
        with open(path, "w") as file:
            json.dump(result, file)


def main():
    algorithms = DataManager.get_algorithms()
    print(algorithms)
    data = DataManager.read_data()
    print(data)


if __name__ == "__main__":
    main()
