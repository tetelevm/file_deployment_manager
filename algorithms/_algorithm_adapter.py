import random

from time_matrix_calculator import TimeMatrix
from ._deployment_matrix import DeploymentMatrix
from ._default_time_matrix import time_matrix_list


__all__ = [
    "BaseAlgorithm",
]


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

    def create_random_matrix(self) -> DeploymentMatrix:
        """
        Creates a random distribution matrix.
        It works similarly to `.create_initial_matrix()`, but it does not
        fill servers one by one, but chooses a random server (among free
        ones) for each file.
        """

        matrix = DeploymentMatrix.null(self.counts["files"], self.counts["sv"])

        server_spaces = self.server_spaces.copy()
        for file_ind in range(matrix.f_count):
            file_weight = self.file_sizes[file_ind]

            available_servers = [
                server_index
                for (server_index, space) in enumerate(server_spaces)
                if space >= file_weight
            ]
            if not available_servers:
                # if for some reason there is no space in the random allocation,
                # then let it not be a random allocation, but definitely available
                return self.create_initial_matrix()

            server_index = random.choice(available_servers)
            server_spaces[server_index] -= file_weight

            matrix[file_ind, server_index] = 1

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
        Just executes its method `.do_one_step()` until
        `self.stop_condition() == False`.
        """

        self.do_one_step()
        while not self.stop_condition():
            self.do_one_step()
        return self.matrix

    # === functions to describe the logic ==============================

    def stop_condition(self) -> bool:
        """
        This function should contain the condition when the algorithm
        stops. If `True` is returned, the algorithm stops, otherwise it
        continues running.
        """
        return True

    def do_one_step(self):
        """
        A method that is called during algorithm execution.
        The concrete implementation defined by the algorithm.
        """
        pass
