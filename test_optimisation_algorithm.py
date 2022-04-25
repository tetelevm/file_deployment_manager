"""
This file has nothing to do with the project, it is just a tool to test
the correctness of the realization of algorithms.
"""

import random
from abc import ABC, abstractmethod
from typing import Iterable
from tkinter import Tk, Canvas


# travelling salesman problem and drawing of the result ================

class Point:
    def __init__(self, number: int, x_min=20, x_max=520, y_min=20, y_max=520):
        self.number = number
        self.x = random.randint(x_min, x_max)
        self.y = random.randint(y_min, y_max)

    @property
    def draw_coordinates(self) -> tuple[int, int, int, int]:
        return (self.x - 2, self.y - 2, self.x + 2, self.y + 2)


class Path:
    def __init__(self, point_number: int, _fake=False):
        self.point_number = point_number

        if _fake:
            # for copying
            return

        self.points = [Point(num) for num in range(point_number)]
        self.order_visits = list(range(point_number))

        # cache shared between all paths
        self.cache_len: dict[tuple[int, int], float] = dict()
        for indexes in self:
            self.update_line_length(indexes)

    def update_line_length(self, indexes: tuple[int, int]):
        if indexes[::-1] in self.cache_len:
            return self.cache_len[indexes[::-1]]

        first_point = self.points[indexes[0]]
        second_point = self.points[indexes[1]]

        delta_x = first_point.x - second_point.x
        delta_y = first_point.y - second_point.y
        length = (delta_x**2 + delta_y**2) ** 0.5

        self.cache_len[indexes] = length
        self.cache_len[indexes[::-1]] = length

    @property
    def length(self) -> float:
        length = 0
        for indexes in self:
            if indexes not in self.cache_len:
                self.update_line_length(indexes)
            length += self.cache_len[indexes]
        return length

    def __iter__(self) -> Iterable[tuple[int, int]]:
        for num in range(self.point_number):
            yield (self.order_visits[num-1], self.order_visits[num])

    def get_lines_coordinate(self) -> Iterable[tuple[int, int, int, int]]:
        for indexes in self:
            yield (
                self.points[indexes[0]].x,
                self.points[indexes[0]].y,
                self.points[indexes[1]].x,
                self.points[indexes[1]].y
            )

    def copy(self):
        path = Path(self.point_number, True)
        path.points = self.points
        path.order_visits = self.order_visits.copy()
        path.cache_len = self.cache_len
        return path


class Tester:
    def __init__(self, algorithm):
        self.algorithm: AlgorithmAbstract = algorithm
        self.line_ids = []

        self.root = Tk()
        self.root.title("Test Algorithm")
        self.canvas = Canvas(
            self.root,
            width=540,
            height=540,
            background="white",
        )
        self.canvas.pack()

    def set_points(self):
        for point in self.algorithm.path.points:
            self.canvas.create_rectangle(*point.draw_coordinates, fill="#f00")

    def create_lines(self):
        self.line_ids = [
            self.canvas.create_line(*line_coordinates)
            for line_coordinates in self.algorithm.path.get_lines_coordinate()
        ]

    def drop_lines(self): 
        for line_id in self.line_ids:
            self.canvas.delete(line_id)

    def run(self):
        self.algorithm.do_one_step()
        self.drop_lines()
        self.create_lines()
        self.canvas.update()
        if not self.algorithm.stop:
            self.canvas.update()
            self.canvas.after(0, self.run)
        else:
            self.root.title("Test Algorithm (stop)")
            self.root.update()
            print("stop")


# algorithm ============================================================


class AlgorithmAbstract(ABC):
    """
    Adapter class between the task and the implementation of the
    algorithm.
    """

    def __init__(self, point_number=100):
        self.path = Path(point_number)
        self.value = self.path.length
        self.stop = False

    def swap_points(self) -> Path:
        new_path = self.path.copy()
        points = new_path.order_visits

        i, j = random.sample(range(self.path.point_number), k=2)
        points[i], points[j] = points[j], points[i]

        return new_path

    def move_point(self) -> Path:
        new_path = self.path.copy()
        i, j = random.sample(range(self.path.point_number), k=2)
        new_path.order_visits.insert(j, new_path.order_visits.pop(i))
        return new_path

    def make_change(self):
        change_funcs = [self.swap_points, self.move_point]
        for func in change_funcs:
            new_path = func()
            new_length = new_path.length
            replace_path = self.make_decision(new_length)
            if replace_path:
                self.path = new_path
                self.value = new_length

    @abstractmethod
    def make_decision(self, new_len: float) -> bool:
        pass

    @abstractmethod
    def do_one_step(self):
        pass


class SimulatedAnnealing(AlgorithmAbstract):
    def __init__(self, point_number=100):
        super().__init__(point_number)
        self.temperature = 10.
        self.cooling_coefficient = 1 - (1e-01 / point_number**1.2)

    def make_decision(self, new_len: float) -> bool:
        if self.value > new_len:
            return True
        if self.temperature > 0.001:
            delta = (self.value - new_len) / self.value * 100
            return random.random() < 2.71 ** (delta / self.temperature)
        return False

    def do_one_step(self):
        self.make_change()
        print(f"{format(self.temperature, '.9f'): <10}  ==  {self.value}")
        self.temperature *= self.cooling_coefficient
        if self.temperature < 1.0e-6:
            self.stop = True


def main():
    algorithm = SimulatedAnnealing(50)
    tester = Tester(algorithm)
    tester.set_points()
    tester.run()
    tester.root.mainloop()


if __name__ == '__main__':
    main()
