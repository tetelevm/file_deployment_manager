from matrix_calculator import TimeMatrix


__all__ = [
    "AlgorithmExecutor",
]


RESULT_MATRIX_TYPE = list[list[bool]]


class AlgorithmExecutor:
    """
    A class that simply executes the algorithm passed to it, without
    calculating anything itself.
    """

    def __init__(
            self,
            counts: dict[str, int],
            file_weight: list[float],
            server_prices: list[float],
            server_spaces: list[float],
            time_matrix: TimeMatrix,
            coefficient: float,
    ):
        self.counts = counts
        self.time_matrix = time_matrix
        self.server_spaces = server_spaces
        self.file_weight = file_weight
        self.coefficient = coefficient

        self.prices = [
            # each file
            [
                # each server
                # (weight, b) * (prices, money/mb) / (megabytes_to_bytes, mb/b)
                file_weight[file_index] * server_prices[server_index] / 1048576
                for server_index in range(counts["sv"])
            ]
            for file_index in range(counts["file"])
        ]

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
            for file_i in range(self.counts["file"])
            if deployment_matrix[file_i][server_i]
        )

        total_delivery_time = 0
        for file_i in range(self.counts["file"]):
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
            possible_places = sum(
                deployment_matrix[file_i][server_i]
                for server_i in range(self.counts["sv"])
                for file_i in range(self.counts["file"])
            )
            if not possible_places:
                return False

        for server_i in range(self.counts["sv"]):
            required_space = sum(
                self.file_weight[file_i]
                for file_i in range(self.counts["file"])
                if deployment_matrix[file_i][server_i]
            )
            if required_space > self.server_spaces[server_i]:
                return False

        return True

    def execute_algorithm(self, algorithm: callable) -> RESULT_MATRIX_TYPE:
        pass
