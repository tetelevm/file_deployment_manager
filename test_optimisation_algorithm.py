"""
This file has nothing to do with the project, it is just a tool to test
the correctness of the realization of algorithms.
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from math import log
from operator import attrgetter
from tkinter import Tk, Canvas
from typing import Iterable



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
        self._length = None

        if _fake:
            # for copying
            return

        self.points = [Point(num) for num in range(point_number)]
        self.order_visits = list(range(point_number))

        # cache shared between all paths
        self.cache_len: dict[tuple[int, int], float] = dict()
        for i in self.order_visits:
            for j in range(point_number):
                self.update_line_length(i, j)

    def update_line_length(self, i: int, j: int):
        first_point = self.points[i]
        second_point = self.points[j]

        delta_x = first_point.x - second_point.x
        delta_y = first_point.y - second_point.y
        length = (delta_x**2 + delta_y**2) ** 0.5

        self.cache_len[(i, j)] = length
        self.cache_len[(j, i)] = length

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = sum(self.cache_len[indexes] for indexes in self)
        return self._length

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

    def set_order(self, order: list[int]):
        self._length = None
        self.order_visits = order

    def copy(self) -> Path:
        path = Path(self.point_number, True)
        path.points = self.points
        path.set_order(self.order_visits.copy())
        path.cache_len = self.cache_len
        return path

    def __eq__(self, other: Path) -> bool:
        start_ind = other.order_visits.index(self.order_visits[0])
        other_order = other.order_visits[start_ind:] + other.order_visits[:start_ind]
        return self.order_visits == other_order

    def swap_points(self):
        i, j = random.sample(range(self.point_number), k=2)
        points = self.order_visits
        points[i], points[j] = points[j], points[i]
        self.set_order(self.order_visits)

    def move_point(self):
        i, j = random.sample(range(self.point_number), k=2)
        self.order_visits.insert(j, self.order_visits.pop(i))
        self.set_order(self.order_visits)


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
        self.point_number = point_number
        self.path = Path(point_number)
        self.stop = False

        self._debug_info = []

    def log(self, iteration):
        info = f"{iteration: <10}  ==  {self.path.length: <20}"
        if self._debug_info:
            info += " >| " + " | ".join(str(arg) for arg in self._debug_info)
            self._debug_info = []
        print(info)

    @abstractmethod
    def do_one_step(self):
        pass


class SimulatedAnnealing(AlgorithmAbstract):
    temperature: float
    cooling_coefficient: float

    def __init__(self, point_number=100):
        super().__init__(point_number)
        self.temperature = 10
        self.cooling_coefficient = 1 - (1e-01 / point_number)

    def make_decision(self, new_len: float) -> bool:
        if self.path.length > new_len:
            return True

        delta = (self.path.length - new_len) / self.path.length * 100
        return random.random() < 2.71 ** (delta / self.temperature)

    def make_change(self):
        for point in range(self.point_number):
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
        self.log(format(self.temperature, '.9f'))
        self.temperature *= self.cooling_coefficient
        if self.temperature < 1.0e-3:
            self.stop = True


class GeneticAlgorithm(AlgorithmAbstract):
    child_count: int
    count_best: int
    population: list[Path]

    def __init__(self, point_number=100):
        super().__init__(point_number)

        self.population_number = 0
        self.population_number_max = (point_number // 1.5) ** 2
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
        self.log(self.population_number)
        if self.population_number >= self.population_number_max:
            self.stop = True


class AntColony(AlgorithmAbstract):
    line_pheromones: dict[tuple[int, int], float]

    class Ant:
        order: list[int]
        not_visited: list[int]

        def __init__(self, colony: AntColony):
            self.colony = colony
            self.path = self.colony.path.copy()
            self.order = list(range(self.colony.point_number))

        def choose_point(self) -> int:
            from_point = self.order[-1]
            weights = [
                self.colony.get_weight(from_point, to_point)
                for to_point in self.not_visited
            ]
            point_indexes = range(len(self.not_visited))
            return random.choices(point_indexes, weights=weights, k=1)[0]

        def hit_road(self) -> Path:
            self.not_visited = self.order
            self.order = []
            self.order.append(self.not_visited.pop(-1))

            while self.not_visited:
                next_point_ind = self.choose_point()
                self.order.append(self.not_visited.pop(next_point_ind))

            self.path.set_order(self.order)
            return self.path

    # ====================

    def __init__(self, point_number=100):
        super().__init__(point_number)

        self.scout_number = 0
        self.scout_count = point_number * 5

        self.ant_count = point_number * 3
        self.evaporation_coefficient = min(0.7, 1/log(log(point_number**1.3)))
        self.threshold = 1 / point_number ** 1.7

        self.line_pheromones = {
            (fr, to): 1
            for fr in range(point_number)
            for to in range(fr + 1, point_number)
        }
        self.ants = [self.Ant(self) for _ in range(self.ant_count)]

    def get_weight(self, fr, to) -> float:
        return self.line_pheromones[(fr, to) if fr < to else (to, fr)]

    def update_pheromones(self, paths: list[Path]):
        for key in self.line_pheromones.keys():
            if self.line_pheromones[key] > self.threshold:
                self.line_pheromones[key] *= self.evaporation_coefficient
            if self.line_pheromones[key] < self.threshold:
                self.line_pheromones[key] = self.threshold

        best_length = self.path.length
        deltas = [path.length - best_length for path in paths]
        worst_delta = deltas[-1]
        weights = [(1 - (delta / worst_delta)) ** 2 for delta in deltas]
        paths.insert(0, self.path)
        weights.insert(0, 1)

        for (path, weight) in zip(paths, weights):
            for (fr, to) in path:
                self.line_pheromones[(fr, to) if fr < to else (to, fr)] += weight

    def filter_paths(self, paths: list[Path]) -> list[Path]:
        unique_paths = [self.path]
        for new_path in paths:
            if new_path.length >= self.path.length:
                continue
            for current_path in unique_paths:
                if new_path == current_path:
                    break
            else:
                unique_paths.append(new_path)

        sorted_paths = sorted(unique_paths, key=attrgetter("length"))
        return sorted_paths

    def scout_area(self):
        paths = [ant.hit_road() for ant in self.ants]
        sorted_paths = self.filter_paths(paths)

        if len(sorted_paths) > 1:
            self.path = sorted_paths.pop(0).copy()
            self.update_pheromones(sorted_paths)

    def do_one_step(self):
        self.scout_number += 1
        self.scout_area()
        self.log(self.scout_number)
        if self.scout_number > self.scout_count:
            self.stop = True


def main():
    algorithm = AntColony(100)
    tester = Tester(algorithm)
    tester.set_points()
    tester.run()
    tester.root.mainloop()

    # quite optimal results by the points count (depending on the position,
    # the optimality varies in range of 10%):
    #
    # 20 - 1600-2000
    # 30 - 2200-2500
    # 50 - 3000
    # 70 - 3500
    # 100  - 5000
    # 300 -



if __name__ == "__main__":
    main()
