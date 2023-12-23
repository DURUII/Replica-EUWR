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

        # normalize
        self.L = len(self.workers[0].options)
        for w_i in self.workers:
            for l in range(self.L):
                w_i.options[l].normalize_cost()

        self.P = {i: w_i.options for i, w_i in enumerate(self.workers)}
        self.f = f

        self.eps_first = eps_first

        # n_i(t) -> count for how many times each arm/worker {i} has been learned
        self.n = np.zeros(self.N)

        # \bar{q}_i -> the average empirical quality value (reward) of arm/worker {i}
        self.q_bar = np.zeros(self.N)

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
        self.B -= np.sum([option.cost for option in P_t.values()])
        cardinality = np.array([len(P_t[i].tasks) if i in P_t else 0 for i in range(self.N)])
        mask = np.isin(range(self.N), list(P_t.keys()))
        self.q_bar = np.where(mask, (self.q_bar * self.n + u_ww) / (self.n + cardinality), self.q_bar)
        self.n += cardinality

    def initialize(self) -> None:
        self.budget_exploration = self.B * self.eps_first
        self.budget_exploitation = self.B - self.budget_exploration

        if self.extended:
            observed_e = np.array([self.workers[i].epsilon() for i in range(self.N)])
            for i in range(self.N):
                for l in range(self.L):
                    self.workers[i].options[l].update_cost(self.f, observed_e[i])
        P_t: dict[int, SimpleOption] = {i: w_i.options[0] for i, w_i in enumerate(self.workers)}
        self.update_profile(P_t)

    def compute_ucb_quality(self, P_t: dict[int, SimpleOption]):
        """
        Compute the UCB-based quality for a selection.

        :param P_t: Dictionary mapping worker index to their chosen option.
        :return: UCB-based quality value of the current selection.
        """

        if not P_t:
            return 0

        # \hat{q}_i(t), compute UCB-based quality value for each worker (Equation 10)
        q_hat = self.q_bar + np.sqrt((self.K + 1) * np.log(np.sum(self.n)) / self.n)
        v = np.zeros(self.M)

        # p_i^l is in P_t [ith worker is selected]
        for i, option in P_t.items():
            # j is in M_i^l [task j is selected for ith worker]
            v[option.tasks] = np.maximum(q_hat[i], v[option.tasks])

        # represented in vector form (Equation 11)
        return np.dot(self.w, v)

    def select_winners(self) -> dict[int, SimpleOption]:
        """
        Select a subset of workers with one option each, maximizing the UCB/cost ratio.

        :return: Dictionary mapping selected worker index to their chosen option.
        """
        # NOTE: for every worker i, at most one option l can be selected in each round t
        P_t = {}
        heap = []

        # Iterate until K workers are selected.
        while len(P_t) < self.K:

            # P \ P_t' (Line 6 & 7)
            for i in [ii for ii in range(self.N) if ii not in P_t]:
                for l, option in enumerate(self.workers[i].options):
                    # Compute UCB quality difference (Equation 12)
                    ucb_diff = self.compute_ucb_quality(P_t | {i: option}) - self.compute_ucb_quality(P_t)
                    criterion = - ucb_diff / option.cost
                    heapq.heappush(heap, (criterion, i, l))

            # Recruit the worker with the maximum ratio of marginal UCB to cost. (Line 7 & 8)
            _, i_star, l_star = heapq.heappop(heap)
            P_t[i_star] = self.workers[i_star].options[l_star]
            heap = []

        return P_t

    def run(self):
        # exploration
        while True:
            # select K randomly
            self.tau += 1
            if self.extended:
                observed_e = np.array([self.workers[i].epsilon() for i in range(self.N)])
                for i in range(self.N):
                    for l in range(self.L):
                        self.workers[i].options[l].update_cost(self.f, observed_e[i])

            P_t = {
                i: random.choice(self.workers[i].options)
                for i in random.sample(range(self.N), k=self.K)
            }

            if self.B - sum(option.cost for option in P_t.values()) <= self.budget_exploitation:
                break
            self.update_profile(P_t)

        # exploitation
        P_t = self.select_winners()
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
