from __future__ import annotations

import random
from dataclasses import dataclass
from operator import attrgetter
from types import MethodType

from ._deployment_matrix import DeploymentMatrix
from ._algorithm_adapter import BaseAlgorithm


__all__ = [
    "BeesColony",
]


@dataclass
class Source:
    """
    The class simply stores one of the solutions and the `nectar`
    associated with it - the number of times the employee bees will fly
    to that source.
    """

    matrix: DeploymentMatrix
    nectar: int


class Bee:
    """
    A class for the single bee.
    The bee can be one of three roles:
    - scout: just looking for a random possible solution
    - onlooker: looking for a solution near the best (of those available)
    - employee: collects `nectar` from the best solution and remembers
        the solution near it
    """

    colony: BeesColony
    source: Source
    fly: MethodType
    types = ["scout", "onlooker", "employee"]

    def __init__(self, type: str, colony: BeesColony):
        self.colony = colony
        self.fly = self._get_fly_func(type)

    def _fly_as_scout(self):
        """
        It just creates a random solution and remembers it.
        This is needed in case good sources get depleted, so that the
        algorithm doesn't get stuck in local minimums.
        """

        self.source = self.colony.get_random_source()

    def _fly_as_onlooker(self):
        """
        Looks for a solution near the current one and does NOT collect
        `nectar` from it. This is so that the best solution doesn't
        deplete too quickly and the solutions near it have time to be
        inspected.
        Must fly out before the employee bees do (because otherwise
        onlookers will use the worst solution instead of the best one).
        """

        best_source = self.colony.get_best_source()
        self.source = self.colony.get_nearby_source(best_source)

    def _fly_as_employee(self):
        """
        Collects `nectar' from the current solution and remembers the
        solution near it.
        This gradually finds solutions that are in the local minimums
        near the current one.
        """

        best_source = self.colony.get_best_source()
        best_source.nectar -= 1
        self.source = self.colony.get_nearby_source(best_source)

    def _get_fly_func(self, type: str):
        """
        Choose the flying function depending on the type of bee.
        """

        fly_as_type = {
            "scout": self._fly_as_scout,
            "onlooker": self._fly_as_onlooker,
            "employee": self._fly_as_employee,
        }
        return fly_as_type[type]


class BeesColony(BaseAlgorithm):
    """
    A class that realizes the bee colony algorithm.

    The idea of the algorithm is this: there is a colony of bees and a
    field of possible sources.
    The source is just a possible solution with an additional parameter
    `nectar`.
    When there is no nectar in the source, the source is forgotten.
    Bees are units of one of the three types that can fly: inspect this
    field of sources and collect (decrease) their nectar gradually.
    After a bee has visited a source, the bee remembers a place near that
    source and tells it to the others.
    Each time it is flying, the bees choose the best one among the active
    sources (which has nectar).

    The optimizing idea is that the bees gradually inspect the area
    around the good solutions, each time leaving only the more profitable
    ones.
    Decreasing nectar keeps the bees from stopping at one solution and
    forces them to definitely inspect others.
    Also, some bees just randomly fly around the field looking for
    random solutions (for a case where all known solutions have already
    been inspected).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        all_count = self.matrix.f_count * self.matrix.sv_count

        self.scout_count = all_count // 2
        self.onlooker_count = all_count // 5
        self.employed_count = all_count // 1
        self.bee_count = self.scout_count + self.onlooker_count + self.employed_count

        self.flying_number = 0
        self.source_count = self.bee_count
        self.nectar = self.employed_count * 4
        self.max_change = 4
        self.decrement_counter = int((all_count ** 0.5) * 3)  # it must be `int`

        # type order should be like this
        bees_counts = (
            ("scout", self.scout_count),
            ("onlooker", self.onlooker_count),
            ("employee", self.employed_count),
        )
        self.bees = [
            Bee(type, colony=self)
            for (type, count) in bees_counts
            for _ in range(count)
        ]
        self.sources = []

    def get_random_source(self) -> Source:
        """
        Returns a new source with a random solution.
        """
        return Source(self.create_random_matrix(), self.nectar)

    def get_best_source(self) -> Source:
        """
        Returns the most profitable known source that is still available
        (nectar is not depleted).
        """

        for source in self.sources:
            if source.nectar:
                return source

        # if there are no more sources, return a random solution
        return self.get_random_source()

    def get_nearby_source(self, source: Source) -> Source:
        """
        Creates a new source (with maximum `nectar`) near the current one.
        Distance from the current source depends on the parameter
        `max_change`, which decreases with time.
        """

        nearby_matrix = source.matrix.copy()
        for _ in range(self.max_change):
            func = random.choice([
                nearby_matrix.change_existence,
                nearby_matrix.swap_existence,
            ])
            func()
        return Source(nearby_matrix, self.nectar)

    # ===

    @property
    def log_params(self):
        return self.flying_number

    def stop_condition(self):
        """
        If the bees are no longer looking for nearby sources, it's time
        to stop working.
        """
        return not self.max_change

    def do_one_step(self):
        """
        Doing one step of the algorithm.
        To do this:
        - lets all the bees fly
        - filters the found sources by solution availability
        - sorts the remaining sources and saves the best one
        - leaves only the sources that have not yet been depleted
        - if a lot of time has passed, it decreases the `flying_number`
            parameter, so that the next nearby sources are looked for
            nearer to the original one
        """

        for bee in self.bees:
            bee.fly()

        all_sources = list(filter(
            lambda s: self.check_prerequisite(s.matrix),
            self.sources + [bee.source for bee in self.bees]
        ))

        sorted_sources = sorted(
            all_sources,
            key=lambda s: self.get_deployment_result(s.matrix)
        )
        if self.get_deployment_result(sorted_sources[0].matrix) < self.best_value:
            self.matrix = sorted_sources[0].matrix.copy()
            self.best_value = self.get_deployment_result()

        active_sources = list(filter(attrgetter("nectar"), sorted_sources))
        self.sources = active_sources[:self.source_count]

        self.flying_number += 1
        if not self.flying_number % self.decrement_counter:
            self.max_change -= 1
