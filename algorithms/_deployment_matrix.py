from __future__ import annotations

import random
from copy import deepcopy


__all__ = [
    "DeploymentMatrix",
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

    def __eq__(self, other: DeploymentMatrix):
        return self.matrix == other.matrix

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

    def change_existence(self):
        """
        Randomly changes the location of one file on one server.
        """

        f_ind = random.randrange(self.f_count)
        s_ind = random.randrange(self.sv_count)
        self[f_ind, s_ind] ^= 1

    def swap_existence(self):
        """
        Randomly moves one file to another server.
        """

        f = random.randrange(self.f_count)
        s_1, s_2 = random.sample(range(self.sv_count), k=2)
        self[f, s_1], self[f, s_2] = self[f, s_2], self[f, s_1]
