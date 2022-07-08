"""
Project master file. Connects all the other parts together, implementing
this algorithm:

- parse configs
- parse of input data
- time matrix calculation
- calculation of the result using the desired algorithms
- recording results in a file
"""


from data_manager import DataManager
from time_matrix_calculator import MatrixCalculator


def main():
    # terrible codestyle, but what to do
    (counts, file_sizes, delays, pc_to_ls, ls_to_cs, cs_to_sv, server_prices,
     server_spaces, coefficient) = DataManager.read_data()
    algorithms = DataManager.get_algorithms()

    matrix_calculator = MatrixCalculator(counts, file_sizes, delays, pc_to_ls,
                                         ls_to_cs, cs_to_sv)
    result = {}
    time_matrix = matrix_calculator.calculate()
    for algorithm_name in algorithms:
        print(f"Running the algorithm <{algorithm_name}>")
        algorithm = DataManager.available_algorithms[algorithm_name]
        calculator = algorithm(counts, file_sizes, server_prices, server_spaces,
                               time_matrix, coefficient)
        matrix = calculator.calculate()
        result[algorithm_name] = matrix.matrix
    DataManager.write_result(result)


if __name__ == "__main__":
    main()
