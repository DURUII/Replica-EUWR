"""
Author: DURUII
Date: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/opt.py
"""
import heapq

import numpy as np

from algorithms.base import BaseAlgorithm
import random

from stakeholder import Worker, Task, SimpleOption


class Opt(BaseAlgorithm):
    def __init__(self, workers: list[Worker], tasks: list[Task], n_selected: int, budget: float, f, extended=False):
        super().__init__(workers, tasks, n_selected, budget)

        self.w = np.array([self.tasks[j].w for j in range(self.M)])

        self.L = len(self.workers[0].options)

        self.P = {i: w_i.options for i, w_i in enumerate(self.workers)}
        self.f = f
        self.extended = extended
        self.P_t = {}

    def compute_utility(self, P_t: dict[int, SimpleOption]):
        u_ww = np.zeros(self.N)
        u_tt = np.zeros(self.M)
        for i, option in P_t.items():
            q_i = np.array([self.workers[i].draw() for _ in range(len(option.tasks))])
            u_tt[option.tasks] = np.maximum(u_tt[option.tasks], q_i)
            u_ww = np.sum(q_i)
        return u_tt, u_ww

    def compute_utility_e(self, P_t: dict[int, SimpleOption]):
        u_ww = np.zeros(self.N)
        u_tt = np.zeros(self.M)
        for i, option in P_t.items():
            q_i = np.array([self.workers[i].mu_q for _ in range(len(option.tasks))])
            u_tt[option.tasks] = np.maximum(u_tt[option.tasks], q_i)
            u_ww = np.sum(q_i)
        return u_tt, u_ww

    def update_profile(self, P_t: dict[int, SimpleOption]):
        u_tt, u_ww = self.compute_utility(P_t)
        self.U += np.dot(self.w, u_tt)
        self.B -= np.sum([option.cost for option in P_t.values()])

    def initialize(self):
        """ steepest ascent """
        P_t = {}
        heap = []
        while len(P_t) < self.K:
            base = np.dot(self.w, self.compute_utility_e(P_t)[0])
            for i in [ii for ii in range(self.N) if ii not in P_t]:
                for l, option in enumerate(self.workers[i].options):
                    u_diff = np.dot(self.w, self.compute_utility_e(P_t | {i: option})[0]) - base
                    criterion = - u_diff / option.cost
                    heapq.heappush(heap, (criterion, i, l))

            _, i_star, l_star = heapq.heappop(heap)
            P_t[i_star] = self.workers[i_star].options[l_star]
            heap = []
        self.P_t = P_t

    def run(self):
        """ selects the top K arms according to r/b every round """
        P_t = self.P_t
        while True:
            self.tau += 1
            if self.extended:
                observed_e = np.array([self.workers[i].epsilon() for i in range(self.N)])
                for i in range(self.N):
                    for l in range(self.L):
                        self.workers[i].options[l].update_cost(self.f, observed_e[i])

            if sum(option.cost for option in P_t.values()) >= self.B:
                break
            self.update_profile(P_t)

        return self.U, self.tau
