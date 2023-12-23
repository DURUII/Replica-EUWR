"""
Author: DURUII
Date: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/eps.py
"""

import numpy as np

from algorithms.base import BaseAlgorithm
import random

from stakeholder import Worker, Task, SimpleOption


class Random(BaseAlgorithm):
    def __init__(self, workers: list[Worker], tasks: list[Task], n_selected: int, budget: float, f):
        super().__init__(workers, tasks, n_selected, budget)

        self.w = np.array([self.tasks[j].w for j in range(self.M)])

        # normalize
        self.L = len(self.workers[0].options)
        for w_i in self.workers:
            for l in range(self.L):
                w_i.options[l].normalize_cost()

        self.P = {i: w_i.options for i, w_i in enumerate(self.workers)}
        self.f = f

    def compute_utility(self, P_t: dict[int, SimpleOption]):
        u_ww = np.zeros(self.N)
        u_tt = np.zeros(self.M)
        for i, option in P_t.items():
            q_i = np.array([self.workers[i].draw() for _ in range(len(option.tasks))])
            u_tt[option.tasks] = np.maximum(u_tt[option.tasks], q_i)
            u_ww = np.sum(q_i)
        return u_tt, u_ww

    def update_profile(self, P_t: dict[int, SimpleOption]):
        u_tt, u_ww = self.compute_utility(P_t)
        self.U += np.dot(self.w, u_tt)
        self.B -= np.sum([option.cost for option in P_t.values()])

    def initialize(self) -> None:
        pass

    def run(self):
        while True:
            self.tau += 1
            P_t = {
                i: random.choice(self.workers[i].options)
                for i in random.sample(range(self.N), k=self.K)
            }
            if sum(option.cost for option in P_t.values()) >= self.B:
                break
            self.update_profile(P_t)
        return self.U, self.tau
