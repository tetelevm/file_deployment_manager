from __future__ import annotations
from copy import deepcopy

def _import_tm_class():
    module_path = Path(__file__).parent.parent.absolute() / "time_matrix_calculator.py"
    spec = importlib.util.spec_from_file_location("tmcpy", module_path)
    tmcpy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tmcpy)
    return tmcpy.TimeMatrix


def _import_tm_list():
    path_path = Path(__file__).parent.absolute() / "_default_time_matrix.py"
    spec = importlib.util.spec_from_file_location("dtmpy", path_path)
    dtmpy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dtmpy)
    return dtmpy.time_matrix_list


try:
    from time_matrix_calculator import TimeMatrix
    from ._default_time_matrix import time_matrix_list
except ModuleNotFoundError:
    # if run as "__main__"
    from pathlib import Path
    import importlib.util
    TimeMatrix = _import_tm_class()
    time_matrix_list = _import_tm_list()


__all__ = [
    "BaseAlgorithm",
    "DeploymentMatrix",
    "abstract_main",
]


class DeploymentMatrix:
    """
    The class that stores the file placement matrix on the servers.
    """

    def __init__(self, matrix: list[list[int]]):
        self.matrix = matrix
        self.f_count = len(matrix)
        self.sv_count = len(matrix[0])
        self.value = None

    def __getitem__(self, index: tuple[int, int]) -> int:
        file, server = index
        return self.matrix[file][server]

    def __setitem__(self, index: tuple[int, int], value: int):
        if self.value is not None:
            self.value = None
        file, server = index
        self.matrix[file][server] = value

    def copy(self) -> DeploymentMatrix:
        """
        Copies itself and returns a new matrix object.
        """
        new_matrix = DeploymentMatrix(deepcopy(self.matrix))
        new_matrix.value = self.value
        return new_matrix

    @classmethod
    def null(cls, f_size, sv_size) -> DeploymentMatrix:
        """
        Creates an empty matrix of size FxSV.
        """
        return cls([[0] * sv_size for _ in range(f_size)])


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

    counts: dict[str, int]
    matrix: DeploymentMatrix
    stop: bool
    best_value: float

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
        self.best_value = self.get_deployment_result()
        self.print_logs = print_logs

    def create_initial_matrix(self) -> DeploymentMatrix:
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

        The method has a couple of hard-to-fix bugs that show up in
        extreme cases (e.g., when space is low and it requires
        non-sequential file locations, or when not all computers are
        connected to all servers).
        """

        # [[X for each server] for each file]
        matrix = DeploymentMatrix.null(self.counts["files"], self.counts["sv"])

        server_ind = 0
        server_space = self.server_spaces[server_ind]
        for file_ind in range(matrix.f_count):
            file_weight = self.file_sizes[file_ind]

            while server_space < file_weight:
                # get first server with free space more than the file weight
                server_ind += 1
                if server_ind >= matrix.sv_count:
                    raise ValueError("Space on servers is less than the files size")
                server_space = self.server_spaces[server_ind]

            matrix[file_ind, server_ind] = 1
            server_space -= file_weight

        return matrix

    def get_deployment_result(self, deployment_matrix: DeploymentMatrix = None) -> float:
        """
        Calculates the cost of placing the files, and then calculates
        the delivery time of all files to all computers and translates
        it into a cost by `coefficient`, and returns the sum of the
        costs.
        """

        if deployment_matrix is None:
            deployment_matrix = self.matrix

        if deployment_matrix.value is None:
            deployment_price = sum(
                self.prices[file_i][server_i]
                for server_i in range(deployment_matrix.sv_count)
                for file_i in range(deployment_matrix.f_count)
                if deployment_matrix[file_i, server_i]
            )
            total_delivery_time = 0
            for file_i in range(deployment_matrix.f_count):
                for server_i in range(deployment_matrix.sv_count):
                    if not deployment_matrix[file_i, server_i]:
                        continue
                    for pc_i in range(self.counts["pc"]):
                        total_delivery_time += self.time_matrix[file_i, server_i, pc_i]
            deployment_matrix.value = deployment_price + self.coefficient * total_delivery_time
        return deployment_matrix.value

    def check_prerequisite(self, deployment_matrix: DeploymentMatrix) -> bool:
        """
        Checks that the prerequisites are met, more precisely:
            - it is possible to get any file from any computer
            - all servers have space for located files
        """

        for pc_i in range(self.counts["pc"]):
            for file_i in range(deployment_matrix.f_count):
                possible_places = sum(
                    self.time_matrix[file_i, server_i, pc_i]
                    for server_i in range(deployment_matrix.sv_count)
                    if deployment_matrix[file_i, server_i]
                )
                if not possible_places:
                    return False

        for server_i in range(deployment_matrix.sv_count):
            required_space = sum(
                self.file_sizes[file_i]
                for file_i in range(deployment_matrix.f_count)
                if deployment_matrix[file_i, server_i]
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

    def do_one_step(self):
        """
        A method that is called during algorithm execution.
        It is obligatory to set `self.stop = True` at the end of its
        work, because the method is called `while not algorithm.stop`.
        The concrete implementation defined by the algorithm.
        """
        self.stop = True


def get_test_data():
    """
    A function that outputs test data to check the workability of the
    algorithms.
    """

    counts = {"files": 20,  "pc": 20, "ls": 15, "cs": 7, "sv": 10}

    file_sizes = [
        56623104, 122683392, 90177536, 93323264, 81788928, 24117248, 50331648,
        20971520, 75497472, 10485760, 0, 94371840, 73400320, 62914560, 36700160,
        110100480, 14680064, 51380224, 42991616, 115343360,
    ]

    server_prices = [40, 100, 50, 30, 70, 20, 60, 10, 90, 80]
    server_spaces = [838860800] * 10  # 800 mb

    time_matrix = TimeMatrix(time_matrix_list)
    coefficient = 1

    return counts, file_sizes, server_prices, server_spaces, time_matrix, coefficient


def abstract_main(algorithm):
    args = get_test_data()
    alg = algorithm(*args, print_logs=True)

    first_result = alg.best_value
    alg.calculate()
    second_result = alg.best_value

    print(alg.matrix)
    print(f"{first_result} => {second_result}")


if __name__ == "__main__":
    abstract_main(BaseAlgorithm)
