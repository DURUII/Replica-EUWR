"""
Author: DURUII
Date: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/opt.py
"""

import numpy as np

from algorithms.base import BaseAlgorithm
import random

from stakeholder import Worker, Task, SimpleOption


class Opt(BaseAlgorithm):
    def __init__(self, workers: list[Worker], tasks: list[Task], n_selected: int, budget: float, f, extended=False):
        super().__init__(workers, tasks, n_selected, budget)

        self.w = np.array([self.tasks[j].w for j in range(self.M)])

        # normalize
        self.L = len(self.workers[0].options)
        for w_i in self.workers:
            for l in range(self.L):
                w_i.options[l].normalize_cost()

        self.P = {i: w_i.options for i, w_i in enumerate(self.workers)}
        self.f = f
        self.extended = extended

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

    def initialize(self):
        """ sort N arms in descending order of p.p.r """
        self.workers.sort(key=lambda w: w.mu_q / w.mu_e, reverse=True)


    def run(self):
        """ selects the top K arms according to r/b every round """
        while True:
            self.tau += 1
            if self.extended:
                observed_e = np.array([self.workers[i].epsilon() for i in range(self.N)])
                for i in range(self.N):
                    for l in range(self.L):
                        self.workers[i].options[l].update_cost(self.f, observed_e[i])
            P_t = {i: random.choice(self.workers[i].options)
                   for i in range(self.K)}
            if sum(option.cost for option in P_t.values()) >= self.B:
                break
            self.update_profile(P_t)

        return self.U, self.tau
