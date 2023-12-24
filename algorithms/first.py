"""
Author: DURUII
Date: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/eps.py
"""
import heapq

import numpy as np

from algorithms.base import BaseAlgorithm
import random

from stakeholder import Worker, Task, SimpleOption


class EpsilonFirst(BaseAlgorithm):
    def __init__(self, workers: list[Worker], tasks: list[Task], n_selected: int, budget: float, f, eps_first: float,
                 extended=False):
        super().__init__(workers, tasks, n_selected, budget)

        self.w = np.array([self.tasks[j].w for j in range(self.M)])

        
        self.L = len(self.workers[0].options)

        self.P = {i: w_i.options for i, w_i in enumerate(self.workers)}
        self.f = f

        self.eps_first = eps_first

        self.n = np.zeros(self.N)
        self.q_bar = np.zeros(self.N)
        self.m = np.zeros(self.N)
        self.eps_bar = np.zeros(self.N)

        # Extra Modification
        self.observed_e = []
        self.emo = [w_i.mu_e for w_i in self.workers]

        # placeholder
        self.budget_exploration = None
        self.budget_exploitation = None
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
        cardinality = np.array([len(P_t[i].tasks) if i in P_t else 0 for i in range(self.N)])
        mask = np.isin(range(self.N), list(P_t.keys()))
        self.q_bar = np.where(mask, (self.q_bar * self.n + u_ww) / (self.n + cardinality), self.q_bar)
        self.n += cardinality
        self.B -= np.sum([option.cost for option in P_t.values()])
        if self.extended:
            self.eps_bar = np.where(mask, (self.eps_bar * self.m + self.observed_e) / (self.eps_bar + 1), self.eps_bar)
            self.m += mask

    def initialize(self) -> None:
        self.budget_exploration = self.B * self.eps_first
        self.budget_exploitation = self.B - self.budget_exploration

        if self.extended:
            self.observed_e = np.array([self.workers[i].epsilon() for i in range(self.N)])
            for i in range(self.N):
                for l in range(self.L):
                    self.workers[i].options[l].update_cost(self.f, self.observed_e[i])
        P_t: dict[int, SimpleOption] = {i: w_i.options[0] for i, w_i in enumerate(self.workers)}
        self.update_profile(P_t)

    def run(self):
        # exploration
        while True:
            # select K randomly
            self.tau += 1
            if self.extended:
                self.observed_e = np.array([self.workers[i].epsilon() for i in range(self.N)])
                for i in range(self.N):
                    for l in range(self.L):
                        self.workers[i].options[l].update_cost(self.f, self.observed_e[i])

            P_t = {
                i: random.choice(self.workers[i].options)
                for i in random.sample(range(self.N), k=self.K)
            }

            if self.B - sum(option.cost for option in P_t.values()) <= self.budget_exploitation:
                break
            self.update_profile(P_t)

        # exploitation
        arrange = np.argsort(self.q_bar / self.emo)[::-1][:self.K]
        if self.extended:
            arrange = np.argsort(self.q_bar / self.eps_bar)[::-1][:self.K]
        while True:
            self.tau += 1

            if self.extended:
                self.observed_e = np.array([self.workers[i].epsilon()
                                      for i in range(self.N)])
                for i in range(self.N):
                    for l in range(self.L):
                        self.workers[i].options[l].update_cost(
                            self.f, self.observed_e[i])

            P_t = {i: random.choice(self.workers[i].options)
                   for i in arrange}

            if sum(option.cost for option in P_t.values()) >= self.B:
                break

            self.update_profile(P_t)

        return self.U, self.tau
