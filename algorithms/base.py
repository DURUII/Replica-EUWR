"""
Author: DURUII
Date: 2023/12/18

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/base.py
2. "II. SYSTEM MODEL & PROBLEM" of the paper
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from stakeholder import Worker, Task


class BaseAlgorithm(metaclass=ABCMeta):
    def __init__(self, workers: list[Worker], tasks: list[Task], n_selected: int, budget: float):
        # INPUTS
        # \mathcal{N}, the set of workers, indexed by i
        self.workers = workers
        self.N = len(workers)

        # \mathcal{M}, the set of tasks, indexed by j
        self.tasks = tasks
        self.M = len(tasks)

        # K, the number of workers recruited in each round
        self.K = n_selected

        # the requesters' budget
        self.B = budget

        # OUTPUTS
        # total achieved weighted completion quality
        self.U = 0.0

        # index for rounds
        self.tau = 1

    @abstractmethod
    def initialize(self) -> None:
        """ initialization """
        pass

    @abstractmethod
    def run(self) -> None:
        """ while-loop """
        pass
