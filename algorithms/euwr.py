"""
Author: DURUII
Date: 2023/12/19

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/aucb.py
2. pseudocode and the corresponding equation definition in the paper
"""

import numpy as np

from algorithms.base import BaseAlgorithm
from stakeholder import ExtendedWorker, Task, Option, SimpleOption, Worker


class EUWR(BaseAlgorithm):
    def __init__(self, workers: list[ExtendedWorker], tasks: list[Task], ratio_selected: int, budget: float, f):
        """
        Initializes the EUWR algorithm.

        :param workers: N
        :param tasks: M
        :param ratio_selected: K = ratio * N
        :param budget: B
        :param f: cost = f(|M_i^l|) * eps
        """

        n_selected = int(ratio_selected * len(workers))
        super().__init__([], tasks, n_selected, budget)
        self.workers: list[ExtendedWorker] = workers

        self.w = np.array([self.tasks[j].w for j in range(self.M)])
        self.w = self.w / sum(self.w)

        self.P = {i: w_i.options for i, w_i in enumerate(workers)}
        self.L = len(workers[0].options)

        self.n = np.zeros(self.N)
        self.q_bar = np.zeros(self.N)

        # Update for extended problem:
        self.f = f
        self.m = np.zeros(self.N)
        self.eps = None
        self.eps_bar = np.zeros(self.N)

    def compute_utility(self, P_t: dict[int, SimpleOption]):
        """
        Compute utility gain in the current round for both worker and task.

        :param P_t: Dictionary mapping worker index to their chosen option.
        :return: Tuple of two arrays: utility of tasks (u_tt) and utility of workers (u_ww).
        """

        u_ww = np.zeros(self.N)
        u_tt = np.zeros(self.M)
        for i, option in P_t.items():
            q_i = np.array([self.workers[i].draw() for _ in range(len(option.tasks))])
            u_tt[option.tasks] = np.maximum(u_tt[option.tasks], q_i)
            u_ww = np.sum(q_i)
        return u_tt, u_ww

    def update_profile(self, P_t: dict[int, SimpleOption]):
        """
        Update the profile of utility, budget, quality, and count of choices after each round.

        :param P_t: Dictionary mapping worker index to their chosen option.
        """

        # Update
        self.eps = np.array([self.workers[i].epsilon() if i in P_t else 0 for i in range(self.N)])
        u_tt, u_ww = self.compute_utility(P_t)
        self.U += np.dot(self.w, u_tt)
        cardinality = np.array([len(P_t[i].tasks) if i in P_t else 0 for i in range(self.N)])
        mask = np.isin(range(self.N), list(P_t.keys()))
        self.q_bar = np.where(mask, (self.q_bar * self.n + u_ww) / (self.n + cardinality), self.q_bar)
        self.n += cardinality

        # Update for extended problem:
        self.B -= np.sum(self.eps * self.f(cardinality))
        self.eps_bar = np.where(mask, (self.eps_bar * self.m + self.eps) / (self.eps_bar + 1), self.eps_bar)
        self.m += mask

    def initialize(self):
        """
        Initial recruitment of all workers at the very beginning of the algorithm.
        """

        P_t: dict[int, Option] = {i: w_i.options[0] for i, w_i in enumerate(self.workers)}
        self.update_profile(P_t)

    def compute_ucb_quality(self, P_t: dict[int, SimpleOption]):
        """
        Incorporate both quality and cost UCB in the selection criterion.

        :param P_t: Dictionary mapping worker index to their chosen option.
        :return: UCB-based quality value of the current selection.
        """

        # Update for extended problem:
        Q_t = np.sqrt((self.K + 1) * np.log(np.sum(self.n)) / self.n)  # [N]
        C_t = np.sqrt((self.K + 1) * np.log(self.tau) / self.m)  # [N]
        f = np.array([len(option.tasks) / self.f(len(option.tasks)) for i, option in P_t.items()])  # [N]

        r_hat = f * self.q_bar / self.eps_bar  # [N]
        r_hat += np.max(f) * (Worker.eps_min * Q_t + C_t) / Worker.eps_min ** 2

        v = np.zeros(self.M)  # [M]
        for i, option in P_t.items():
            v[option.tasks] = np.maximum(r_hat[i], v[option.tasks])
        return np.dot(self.w, v)

    def select_winners(self) -> dict[int, SimpleOption]:
        """
        Select a subset of workers with one option each, maximizing the UCB/cost ratio.

        :return: Dictionary mapping selected worker index to their chosen option.
        """
        P_t = {}
        while len(P_t) < self.K:
            items = []
            for i in [ii for ii in range(self.N) if ii not in P_t]:
                for l, option in enumerate(self.workers[i].options):
                    # Select workers based on a new UCB-based criterion that considers the cost
                    ucb_diff = self.compute_ucb_quality(P_t | {i: option}) - self.compute_ucb_quality(P_t)
                    criterion = ucb_diff / option.compute_cost(self.f, self.eps)
                    items.append((i, l, criterion))
            if items:
                i_star, l_star, _ = max(items, key=lambda x: x[2])
                P_t[i_star] = self.workers[i_star].options[l_star]
        return P_t

    def run(self):
        """
        Main loop to run the EUWR algorithm until the budget is exhausted

        :return: Total utility achieved and the number of rounds conducted.
        """
        while True:
            self.tau += 1
            # The algorithm now adjusts for the dynamic cost estimation
            P_t = self.select_winners()
            print(self.tau, self.B)
            if sum(option.compute_cost(self.f, self.eps) for option in P_t.values()) >= self.B:
                break
            self.update_profile(P_t)
        return self.U, self.tau
