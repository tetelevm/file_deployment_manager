"""
This file has nothing to do with the project, it is just a tool to test
the correctness of the realization of algorithms.
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Iterable
from operator import attrgetter
from tkinter import Tk, Canvas


flip_coin = lambda: random.random() > 0.5

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

    def copy(self) -> Path:
        path = Path(self.point_number, True)
        path.points = self.points
        path.order_visits = self.order_visits.copy()
        path.cache_len = self.cache_len
        return path

    def swap_points(self):
        i, j = random.sample(range(self.point_number), k=2)
        points = self.order_visits
        points[i], points[j] = points[j], points[i]

    def move_point(self):
        i, j = random.sample(range(self.point_number), k=2)
        self.order_visits.insert(j, self.order_visits.pop(i))


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
        self.stop = False

    @abstractmethod
    def do_one_step(self):
        pass


class SimulatedAnnealing(AlgorithmAbstract):
    def __init__(self, point_number=100):
        super().__init__(point_number)
        self.temperature = 10.
        self.cooling_coefficient = 1 - (1e-01 / point_number**1.2)

    def make_decision(self, new_len: float) -> bool:
        if self.path.length > new_len:
            return True
        if self.temperature > 0.001:
            delta = (self.path.length - new_len) / self.path.length * 100
            return random.random() < 2.71 ** (delta / self.temperature)
        return False

    def make_change(self):
        new_path = self.path.copy()
        new_path.swap_points()
        if self.make_decision(new_path.length):
            self.path = new_path

        new_path = self.path.copy()
        new_path.move_point()
        if self.make_decision(new_path.length):
            self.path = new_path

    def do_one_step(self):
        self.make_change()
        print(f"{format(self.temperature, '.9f'): <10}  ==  {self.path.length}")
        self.temperature *= self.cooling_coefficient
        if self.temperature < 1.0e-6:
            self.stop = True


class GeneticAlgorithm(AlgorithmAbstract):
    def __init__(self, point_number=100):
        super().__init__(point_number)

        self.population_number = 0
        self.population_number_max = (point_number / 10) ** 2.25 * 20
        self.child_count = 10
        self.count_best = 10
        self.population_count = self.child_count * self.count_best
        self.population = self.create_descendants(self.path, self.population_count)

    @staticmethod
    def create_descendants(parent_path: Path, count: int) -> list[Path]:
        return [parent_path.copy() for _ in range(count)]

    def filter_best_paths(self):
        paths = sorted(self.population + [self.path], key=attrgetter("length"))
        best_paths = paths[:self.count_best]
        self.path = best_paths[0]

        new_population = []
        for path in best_paths:
            child = self.create_descendants(path, self.child_count)
            new_population.extend(child)
        self.population = new_population

    def grow_generation(self):
        # how to crossbreed paths, I have not figured out, so only mutations
        for path in self.population:
            if flip_coin():
                path.swap_points()
            if flip_coin():
                path.move_point()
        self.filter_best_paths()

    def do_one_step(self):
        self.population_number += 1
        self.grow_generation()
        print(f"{self.population_number: <8}  ==  {self.path.length}")
        if self.population_number >= self.population_number_max:
            self.stop = True


def main():
    algorithm = GeneticAlgorithm(100)
    tester = Tester(algorithm)
    tester.set_points()
    tester.run()
    tester.root.mainloop()


if __name__ == '__main__':
    main()
