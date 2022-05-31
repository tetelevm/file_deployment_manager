import random
try:
    from .algorithm_adapter import BaseAlgorithm, get_test_data
except ImportError:
    # if run as "__main__"
    from algorithm_adapter import BaseAlgorithm, get_test_data, abstract_main


__all__ = [
    "SimulatedAnnealing",
]


class SimulatedAnnealing(BaseAlgorithm):
    """
    A class that realizes the annealing algorithm in the context of the
    project.
    It has 3 own variables - temperature, cooling coefficient and
    minimum temperature. The algorithm works until the temperature of
    the algorithm drops to the minimum temperature.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = 100.
        self.minimum_temperature = 1.0e-3

        # coefficient depends on the number of parameters
        all_count = self.counts["files"] * self.counts["sv"] * self.counts["pc"]
        # the function is poorly configured, you need to check on other parameters
        self.cooling_coefficient = 1 - (1.1**all_count) * 1e-04

    def make_decision(self, new_value: float) -> bool:
        """
        Makes a choice whether the matrix fits or not.
        The higher the temperature, the greater the chance that the
        matrix will change, even if it is much worse than it was. At
        very low temperatures, on the contrary, only the values that are
        unambiguously better are taken (you could say it becomes a
        greedy algorithm).
        """

        if self.best_value > new_value:
            return True
        if self.temperature > 0.01:
            # checks the percentage change between the values
            delta = (self.best_value - new_value) / self.best_value * 100
            return random.random() < 2.71 ** (delta / self.temperature)
        return False

    def is_better_than_current(self) -> bool:
        """
        A little additional function, just to avoid duplicating code.
        Checks if the matrix fits the conditions, and if it does, takes
        it as a new matrix with some probability.
        """

        if not self.check_prerequisite(self.matrix):
            return False

        new_value = self.get_deployment_result(self.matrix)
        is_to_set = self.make_decision(new_value)
        if is_to_set:
            self.best_value = new_value

        return is_to_set

    def change_existence(self, f_ind: int, s_ind: int):
        """
        A function that changes the existence of a file (if it doesn't
        fit, it changes back).
        """
        self.matrix[f_ind][s_ind] ^= 1
        if not self.is_better_than_current():
            self.matrix[f_ind][s_ind] ^= 1

    def swap_existence(self, f1: int, s1: int, f2: int, s2: int):
        """
        A function that swaps the existence of two files (if it doesn't
        fit, it swaps back).
        """
        m = self.matrix  # for short record
        m[f1][s1], m[f2][s2] = m[f2][s2], m[f1][s1]
        if not self.is_better_than_current():
            m[f1][s1], m[f2][s2] = m[f2][s2], m[f1][s1]

    def make_change(self):
        """
        Makes one cycle of changes.
        One at a time for each row and column it changes the existence
        of a random file, and then for each row and column it changes
        the existence of two random files.
        """

        server_count = self.counts["sv"]
        file_count = self.counts["files"]

        for file_ind in range(file_count):
            server_ind = random.randrange(server_count)
            self.change_existence(file_ind, server_ind)
        for server_ind in range(server_count):
            file_ind = random.randrange(file_count)
            self.change_existence(file_ind, server_ind)

        for file_ind in range(file_count):
            server_ind_1, server_ind_2 = random.sample(range(server_count), k=2)
            self.swap_existence(file_ind, server_ind_1, file_ind, server_ind_2)
        for server_ind in range(server_count):
            file_ind_1, file_ind_2 = random.sample(range(file_count), k=2)
            self.swap_existence(file_ind_1, server_ind, file_ind_2, server_ind)

    def do_one_step(self):
        """
        Makes one cycle of changes, prints the results and reduces the
        temperature. If the temperature has dropped to a certain value,
        it stops the algorithm.
        """

        self.make_change()
        if self.print_logs:
            print(f"{format(self.temperature, '.9f'): <10}  ==  {self.best_value}")
        self.temperature *= self.cooling_coefficient
        if self.temperature < self.minimum_temperature:
            self.stop = True


if __name__ == '__main__':
    abstract_main(SimulatedAnnealing)
