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
from matrix_calculator import MatrixCalculator


def main():
    # terrible codestyle, but what to do
    (counts, file_sizes, delays, pc_to_ls, ls_to_cs, cs_to_sv, server_prices,
     server_spaces, coefficient) = DataManager.read_data()
    algorithms = DataManager.get_algorithms()

    matrix_calculator = MatrixCalculator(counts, file_sizes, delays, pc_to_ls,
                                         ls_to_cs, cs_to_sv)
    time_matrix = matrix_calculator.calculate()
    for algorithm_name in algorithms:
        algorithm = DataManager.available_algorithms[algorithm_name]
        calculator = algorithm(counts, file_sizes, server_prices, server_spaces,
                               time_matrix, coefficient, print_logs=True)
        calculator.calculate()
        print(calculator.matrix)


if __name__ == "__main__":
    main()
