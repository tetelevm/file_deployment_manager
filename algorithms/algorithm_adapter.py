from abc import abstractmethod

try:
    from matrix_calculator import TimeMatrix
except ModuleNotFoundError:
    # if run as "__main__"
    from pathlib import Path
    import importlib.util
    matrix_path = Path(__file__).parent.parent.absolute() / "matrix_calculator.py"
    spec = importlib.util.spec_from_file_location("matrix_calculator", matrix_path)
    matrix_calculator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix_calculator)
    TimeMatrix = matrix_calculator.TimeMatrix


__all__ = [
    "BaseAlgorithm",
    "RESULT_MATRIX_TYPE",
    "get_test_data",
]


RESULT_MATRIX_TYPE = list[list[int]]


class BaseAlgorithm():
    """
    Adapter class between the file distribution task and the
    implementation of the algorithms. All other classes of algorithms
    are inherited from it.

    Takes all parameters, creates a price matrix and a file distribution
    matrix, and then calculates the optimal distribution.

    The public call points of the class are the `.calculate()` method
    and the `.matrix` attribute obtained after the calculation.
    """

    def __init__(
            self,
            counts: dict[str, int],
            file_sizes: list[float],
            server_prices: list[float],
            server_spaces: list[float],
            time_matrix: TimeMatrix,
            coefficient: float,
            print_logs: bool = False
    ):
        self.counts = counts
        self.time_matrix = time_matrix
        self.server_spaces = server_spaces
        self.file_sizes = file_sizes
        self.coefficient = coefficient

        self.prices = [
            # each file
            [
                # each server
                # (weight, b) * (prices, money/mb) / (megabytes_to_bytes, mb/b)
                file_sizes[file_index] * server_prices[server_index] / 1048576
                for server_index in range(counts["sv"])
            ]
            for file_index in range(counts["files"])
        ]
        self.matrix = self.create_initial_matrix()

        self.stop = False
        self.best_value = self.get_deployment_result(self.matrix)
        self.print_logs = print_logs

    def create_initial_matrix(self) -> list[list[int]]:
        """
        Creates an initial distribution matrix. This matrix is a list
        of lists of `file X server` size ints.
        =====================================
        => files -> 5mb, 5mb, 5mb
        => servers -> 12mb, 15mb, 10mb, 10mb
        >>> [
        >>>     [1, 0, 0, 0],  # ^
        >>>     [1, 0, 0, 0],  # 3 files
        >>>     [0, 1, 0, 0],  # V
        >>>   # < 4 servers >
        >>> ]

        The distribution is calculated rather silly, just filling the
        servers with files one by one, until it runs out of files or
        space on the servers (then an `ValueError` will be caused).
        =====================================
        +------------------------------------------------------------+
        | <-first file-> / <-second file-> / ~third file is too big~ |
        +------------------------------------------------------------+
        | <-third file-> / ~free space~                              |
        +------------------------------------------------------------+
        | ~free space~                                               |
        +------------------------------------------------------------+
        | ~free space~                                               |
        +------------------------------------------------------------+

        A small bug in the calculation - since both file and server are
        taken one by one, then a smaller later file may fit on one of
        the earlier servers, but the calculation will give an error.
        =====================================
        => files -> 5mb, 5mb, 5mb, 5mb, 5mb, 2 mb
        => servers -> 13mb, 10, 6mb

        /            5mb          /          5mb            /      3mb      /
        +-------------------------------------------------------------------+
        |         < first >       /       < second ->       /   free space  |
        +-------------------------------------------------------------------+
        |         < third >       /       < fourth >        |           ^
        +---------------------------------------------------+           |
        |         < fifth >       /     |  < 1 free mb there            |
        +-------------------------------+    but file weighs 2 mb       |
                                  / < sixth >/                          |
                                      in practice this file would fit here
                                      but the script will raise an error

        There may also be errors if the download is not available
        between some computers and servers, as the first initiation (in
        the current version) does not take them into account. For
        example, if a file cannot be downloaded anywhere from the first
        server, the file will be installed there anyway.
        """

        # [[X for each server] for each file]
        matrix = [
            [0] * self.counts["sv"]
            for _ in range(self.counts["files"])
        ]

        server_ind = 0
        server_space = self.server_spaces[server_ind]
        for file_ind in range(self.counts["files"]):
            file_weight = self.file_sizes[file_ind]

            while server_space < file_weight:
                # get first server with free space more than the file weight
                server_ind += 1
                if server_ind >= self.counts["sv"]:
                    raise ValueError("Space on servers is less than the files size")
                server_space = self.server_spaces[server_ind]

            matrix[file_ind][server_ind] = 1
            server_space -= file_weight

        return matrix

    def get_deployment_result(self, deployment_matrix: RESULT_MATRIX_TYPE) -> float:
        """
        Calculates the cost of placing the files, and then calculates
        the delivery time of all files to all computers and translates
        it into a cost by `coefficient`, and returns the sum of the
        costs.
        """

        deployment_price = sum(
            self.prices[file_i][server_i]
            for server_i in range(self.counts["sv"])
            for file_i in range(self.counts["files"])
            if deployment_matrix[file_i][server_i]
        )

        total_delivery_time = 0
        for file_i in range(self.counts["files"]):
            for server_i in range(self.counts["sv"]):
                if not deployment_matrix[file_i][server_i]:
                    continue
                for pc_i in range(self.counts["pc"]):
                    total_delivery_time += self.time_matrix[file_i, server_i, pc_i]

        return deployment_price + self.coefficient * total_delivery_time

    def check_prerequisite(self, deployment_matrix: RESULT_MATRIX_TYPE) -> bool:
        """
        Checks that the prerequisites are met, more precisely:
            - it is possible to get any file from any computer
            - all servers have space for located files
        """

        for pc_i in range(self.counts["pc"]):
            for file_i in range(self.counts["files"]):
                possible_places = sum(
                    self.time_matrix[file_i, server_i, pc_i]
                    for server_i in range(self.counts["sv"])
                    if deployment_matrix[file_i][server_i]
                )
                if not possible_places:
                    return False

        for server_i in range(self.counts["sv"]):
            required_space = sum(
                self.file_sizes[file_i]
                for file_i in range(self.counts["files"])
                if deployment_matrix[file_i][server_i]
            )
            if required_space > self.server_spaces[server_i]:
                return False

        return True

    def calculate(self):
        """
        Calculates until it stops :)
        Just executes its method `.do_one_step()` until it set
        `self.stop = True`.
        """
        while not self.stop:
            self.do_one_step()
        return self.matrix

    @abstractmethod
    def do_one_step(self):
        """
        A method that is called during algorithm execution.
        It is obligatory to set `self.stop = True` at the end of its
        work, because the method is called `while not algorithm.stop`.
        The concrete implementation defined by the algorithm.
        """
        pass


def get_test_data():
    """
    A function that outputs test data to check the workability of the
    algorithms.
    """

    counts = {"files": 10,  "pc": 2, "ls": 3, "cs": 4, "sv": 2}

    byte_on_mb = 2 ** 20
    file_sizes = [
        54 * byte_on_mb, 117 * byte_on_mb, 86 * byte_on_mb, 89 * byte_on_mb,
        78 * byte_on_mb, 23 * byte_on_mb, 48 * byte_on_mb, 20 * byte_on_mb,
        72 * byte_on_mb, 10 * byte_on_mb
    ]

    server_prices = [10, 100]
    server_spaces = [400 * byte_on_mb, 600 * byte_on_mb]

    time_matrix_list = [
        [[0, 0], [5.5, 5.5]],
        [[11.8, 11.8], [11.8, 4.45]],
        [[8.7, 8.7], [8.7, 3.42]],
        [[9.0, 9.0], [9.0, 3.52]],
        [[7.9, 7.9], [7.9, 3.15]],
        [[2.4, 2.4], [2.4, 1.32]],
        [[4.9, 4.9], [4.9, 2.15]],
        [[2.1, 2.1], [2.1, 1.22]],
        [[7.3, 7.3], [7.3, 2.95]],
        [[1.1, 1.1], [1.1, 0.88]],
    ]
    time_matrix = TimeMatrix(time_matrix_list)
    coefficient = 1

    return counts, file_sizes, server_prices, server_spaces, time_matrix, coefficient


def main():
    args = get_test_data()
    alg = BaseAlgorithm(*args)
    check_result = alg.check_prerequisite(alg.matrix)
    print(alg.matrix)
    print(check_result)
    print(alg.best_value)


if __name__ == '__main__':
    main()
