"""
This file has nothing to do with the project, it is just a tool to test
the correctness of the realization of algorithms.
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import log
from operator import attrgetter
from tkinter import Tk, Canvas
from typing import Iterable, Protocol, TypeVar, Any


# === funcs ============================================================


def flip_coin() -> bool:
    return random.random() > 0.5


class _Sortable_V(Protocol):
    length: float


class _Sortable_P(Protocol):
    @property
    def length(self) -> float:
        return 0.


SortableT = TypeVar("SortableT", bound=_Sortable_V | _Sortable_P)


# === travelling salesman problem and drawing of the result ============


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
        for i in range(point_number):
            for j in range(i, point_number):
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

    def shuffle(self):
        random.shuffle(self.order_visits)
        self.set_order(self.order_visits)

    @staticmethod
    def swap_points(path: Path):
        i, j = random.sample(range(path.point_number), k=2)
        points = path.order_visits
        points[i], points[j] = points[j], points[i]
        path.set_order(path.order_visits)

    @staticmethod
    def move_point(path: Path):
        i, j = random.sample(range(path.point_number), k=2)
        path.order_visits.insert(j, path.order_visits.pop(i))
        path.set_order(path.order_visits)

    modify_funcs = [swap_points, move_point]


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
            print(f"result: {self.algorithm.path.length: <20}")


# === algorithms =======================================================


class AlgorithmAbstract(ABC):
    """
    Adapter class between the task and the implementation of the
    algorithm.
    """

    point_number: int
    path: Path
    stop: bool

    def __init__(self, point_number=100, params: dict[str, Any] = None):
        self.point_number = point_number
        self.path = Path(point_number)
        self.stop = False

        self.counter = 0
        self._debug_info = []
        self.log = self._log

        if params is None:
            params = self.get_default_params()
        self.init_from_params_dct(params)
        self.post_init_params()

    def _log(self, *args, _counter: str = None):
        counter = _counter or self.counter
        info = [f"{counter: <6}", f"{self.path.length: <20}"]

        if args:
            info.extend(args)

        if self._debug_info:
            info.extend(self._debug_info)
            self._debug_info = []

        print(" | ".join(str(arg) for arg in info))

        self.counter += 1

    def _no_log(self, *args, **kwargs):
        self._debug_info = []
        self.counter += 1

    def disable_log(self):
        self.log = self._no_log

    @staticmethod
    def sort_paths(paths: list[SortableT]) -> list[SortableT]:
        return sorted(paths, key=attrgetter("length"))

    def init_from_params_dct(self, params: dict[str, Any]):
        for (field, value) in params.items():
            if callable(value):
                setattr(self, field, value(self))
            else:
                setattr(self, field, value)

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def post_init_params(self):
        pass

    @abstractmethod
    def stop_condition(self) -> bool:
        pass

    @abstractmethod
    def algorithm_cycle(self) -> tuple | None:
        pass

    def do_one_step(self):
        args = self.algorithm_cycle() or []
        self.log(*args)
        if self.stop_condition():
            self.stop = True


class SimulatedAnnealing(AlgorithmAbstract):
    temperature: float
    cooling_coefficient: float

    def get_default_params(self):
        return {
            "temperature": 10,
            "cooling_coefficient": 1 - (1e-01 / self.point_number),
        }

    def post_init_params(self):
        pass

    def make_decision(self, new_len: float) -> bool:
        if self.path.length > new_len:
            return True

        delta = (self.path.length - new_len) / self.path.length * 100
        return random.random() < 2.71 ** (delta / self.temperature)

    def algorithm_cycle(self):
        for point in range(self.point_number):
            for modify_func in Path.modify_funcs:
                new_path = self.path.copy()
                modify_func(new_path)
                if self.make_decision(new_path.length):
                    self.path = new_path

        self.temperature *= self.cooling_coefficient
        return (format(self.temperature, '.9f'),)

    def stop_condition(self):
        return self.temperature < 1.0e-3


class GeneticAlgorithm(AlgorithmAbstract):
    population_number_max: float
    child_count: int
    count_best: int
    population_count: int

    population: list[Path]

    def get_default_params(self):
        return {
            "population_number_max": (self.point_number // 1.5) ** 2,
            "child_count": 10,
            "count_best": 10,
            "population_count": lambda s: s.child_count * s.count_best,
        }

    def post_init_params(self):
        self.population = self.create_descendants(self.path, self.population_count)
        for path in self.population:
            path.shuffle()

    @staticmethod
    def create_descendants(parent_path: Path, count: int) -> list[Path]:
        return [parent_path.copy() for _ in range(count)]

    def filter_best_paths(self):
        paths = self.sort_paths(self.population + [self.path])
        best_paths = paths[:self.count_best]
        self.path = best_paths[0]

        new_population = []
        for path in best_paths:
            child = self.create_descendants(path, self.child_count)
            new_population.extend(child)
        self.population = new_population

    def algorithm_cycle(self):
        # how to crossbreed paths, I have not figured out, so only mutations
        for path in self.population:
            for modify_func in Path.modify_funcs:
                if flip_coin():
                    modify_func(path)
        self.filter_best_paths()

    def stop_condition(self):
        return self.counter >= self.population_number_max


class AntColony(AlgorithmAbstract):

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

    scout_count: int
    ant_count: int
    evaporation_coefficient: float
    threshold: float

    line_pheromones: dict[tuple[int, int], float]
    ants: list[Ant]

    def get_default_params(self):
        return {
            "scout_count": self.point_number * 5,
            "ant_count": self.point_number * 3,
            "evaporation_coefficient": min(0.7, 1/log(log(self.point_number**1.3))),
            "threshold": 1 / self.point_number ** 1.7,
        }

    def post_init_params(self):
        self.line_pheromones = {
            (fr, to): 1
            for fr in range(self.point_number)
            for to in range(fr + 1, self.point_number)
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

        sorted_paths = self.sort_paths(unique_paths)
        return sorted_paths

    def algorithm_cycle(self):
        paths = [ant.hit_road() for ant in self.ants]
        sorted_paths = self.filter_paths(paths)

        if len(sorted_paths) > 1:
            self.path = sorted_paths.pop(0).copy()
            self.update_pheromones(sorted_paths)

    def stop_condition(self):
        return self.counter > self.scout_count


class BeesColony(AlgorithmAbstract):

    @dataclass
    class Source:
        path: Path
        nectar: int

        @property
        def length(self) -> float:
            return self.path.length

    class Bee:
        source: BeesColony.Source
        types = ["scout", "onlooker", "employee"]

        def __init__(self, type: str, colony: BeesColony):
            self.colony = colony
            self.fly = self.get_fly_func(type)

        def _fly_as_scout(self):
            self.source = self.colony.get_random_source()

        def _fly_as_onlooker(self):
            best_source = self.colony.get_best_source()
            best_source.nectar -= 1
            self.source = self.colony.get_nearby_source(best_source)

        def _fly_as_employee(self):
            best_source = self.colony.get_best_source()
            self.source = self.colony.get_nearby_source(best_source)
            self.source.nectar -= 1

        def get_fly_func(self, type: str):
            fly_as_type = {
                "scout": self._fly_as_scout,
                "onlooker": self._fly_as_onlooker,
                "employee": self._fly_as_employee,
            }
            return fly_as_type[type]

    # ====================

    scout_count: int
    onlooker_count: int
    employed_count: int
    bee_count: int
    source_count: int
    nectar: int
    max_change: int
    decrement_counter: int

    sources: list[Source]
    bees: list[Bee]

    def get_default_params(self):
        return {
            "scout_count": self.point_number * 1,
            "onlooker_count": self.point_number * 5,
            "employed_count": self.point_number * 10,
            "bee_count": lambda s: s.scout_count + s.onlooker_count + s.employed_count,
            "source_count": lambda s: s.bee_count * 2,
            "nectar": lambda s: s.onlooker_count * 3,
            "max_change": 10,
            "decrement_counter": self.point_number * 4,
        }

    def post_init_params(self):
        self.bees = [
            *(
                self.Bee("scout", colony=self)
                for _ in range(self.scout_count)
            ),
            *(
                self.Bee("onlooker", colony=self)
                for _ in range(self.onlooker_count)
            ),
            *(
                self.Bee("employee", colony=self)
                for _ in range(self.employed_count)
            ),
        ]
        self.sources = []

    def get_random_source(self) -> Source:
        source = BeesColony.Source(self.path.copy(), self.nectar)
        source.path.shuffle()
        return source

    def get_best_source(self) -> Source:
        for source in self.sources:
            if source.nectar:
                return source
        return self.get_random_source()

    def get_nearby_source(self, source: Source) -> Source:
        nearby_path = source.path.copy()
        for _ in range(self.max_change):
            func = random.choice(Path.modify_funcs)
            func(nearby_path)
        return BeesColony.Source(nearby_path, self.nectar)

    def algorithm_cycle(self):
        for bee in self.bees:
            bee.fly()

        all_sources = self.sort_paths(self.sources + [bee.source for bee in self.bees])
        if all_sources[0].length < self.path.length:
            self.path = all_sources[0].path.copy()

        active_sources = list(filter(attrgetter("nectar"), all_sources))
        self.sources = active_sources[:self.source_count]

        if not self.counter % self.decrement_counter:
            self.max_change -= 1

    def stop_condition(self):
        return not self.max_change


# === main =============================================================


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
    # 300 - 9000-10000


if __name__ == "__main__":
    main()
