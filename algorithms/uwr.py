"""
Author: DURUII
Date: 2023/12/19

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/aucb.py
2. pseudocode and the corresponding equation definition in the paper
"""
import numpy as np

from algorithms.base import BaseAlgorithm
from stakeholder import Worker, Task, Option


class UWR(BaseAlgorithm):
    def __init__(self, workers: list[Worker], tasks: list[Task], n_selected: int, budget: float):
        """
        Initialize the UWR algorithm.

        :param workers: List of Worker objects.
        :param tasks: List of Task objects.
        :param n_selected: Number of workers to be selected in each round.
        :param budget: Total budget available for task allocations.
        """

        super().__init__(workers, tasks, n_selected, budget)

        # ADDITIONAL INPUT
        # importance weight of a given task, normalized to sum to one
        self.w = np.array([self.tasks[j].w for j in range(self.M)])
        self.w = self.w / sum(self.w)

        # sorted options for each worker based on cost
        # storage, you can use P[i][l] to access p_i^l = (tasks, cost)
        self.P = {i: sorted(w_i.options, key=lambda o: o.cost) for i, w_i in enumerate(workers)}
        self.L = len(workers[0].options)

        # PROFILE
        # n_i(t) -> count for how many times each arm/worker {i} has been learned
        self.n = np.zeros(self.N)

        # \bar{q}_i -> the average empirical quality value (reward) of arm/worker {i}
        self.q_bar = np.zeros(self.N)

    def compute_utility(self, P_t: dict[int, Option]):
        """
        Compute utility gain in the current round for both worker and task.

        :param P_t: Dictionary mapping worker index to their chosen option.
        :return: Tuple of two arrays: utility of tasks (u_tt) and utility of workers (u_ww).
        """

        u_ww = np.zeros(self.N)  # final completion quality of worker i
        u_tt = np.zeros(self.M)  # that of task j (Equation 1)

        for i, option in P_t.items():
            # with the same length as option.tasks
            q_i = np.array([self.workers[i].draw() for _ in range(len(option.tasks))])
            # Max utility for each task, used to later compute U (Equation 2)
            u_tt[option.tasks] = np.maximum(u_tt[option.tasks], q_i)
            # Total utility for each worker, used to later computer q_bar (Equation 9)
            u_ww = np.sum(q_i)

        return u_tt, u_ww

    def update_profile(self, P_t: dict[int, Option]):
        """
        Update the profile of utility, budget, quality, and count of choices after each round.

        :param P_t: Dictionary mapping worker index to their chosen option.
        """
        # compute u^i(P_t) and u^j(P_t)
        u_tt, u_ww = self.compute_utility(P_t)

        # Add utility in this round to total record and deduct the cost (Line 13)
        self.U += np.dot(self.w, u_tt)
        self.B -= np.sum([option.cost for option in P_t.values()])

        # |M_i^t| if option {l} of worker {i} is selected else 0
        cardinality = np.array([len(P_t[i].tasks) if i in P_t else 0 for i in range(self.N)])
        # whether p_i^l is in P_t
        mask = np.isin(range(self.N), list(P_t.keys()))

        # update average quality value for each worker (Equation 9, Line 12)
        self.q_bar = np.where(mask, (self.q_bar * self.n + u_ww) / (self.n + cardinality), self.q_bar)
        self.n += cardinality

    def initialize(self):
        """
        Initial recruitment of all workers at the very beginning of the algorithm.
        """

        # Select the first option for each worker and update the profile accordingly.
        P_t: dict[int, Option] = {i: w_i.options[0] for i, w_i in enumerate(self.workers)}
        self.update_profile(P_t)

    def compute_ucb_quality(self, P_t: dict[int, Option]):
        """
        Compute the UCB-based quality for a selection.

        :param P_t: Dictionary mapping worker index to their chosen option.
        :return: UCB-based quality value of the current selection.
        """

        # \hat{q}_i(t), compute UCB-based quality value for each worker (Equation 10)
        q_hat = self.q_bar + np.sqrt((self.K + 1) * np.log(np.sum(self.n)) / self.n)
        v = np.zeros(self.M)

        # p_i^l is in P_t [ith worker is selected]
        for i, option in P_t.items():
            # j is in M_i^l [task j is selected for ith worker]
            v[option.tasks] = np.maximum(q_hat[i], v[option.tasks])

        # represented in vector form (Equation 11)
        return np.dot(self.w, v)

    def select_winners(self) -> dict[int, Option]:
        """
        Select a subset of workers with one option each, maximizing the UCB/cost ratio.

        :return: Dictionary mapping selected worker index to their chosen option.
        """
        # NOTE: for every worker i, at most one option l can be selected in each round t
        P_t = {}

        # Iterate until K workers are selected.
        while len(P_t) < self.K:
            items = []

            # P \ P_t' (Line 6 & 7)
            for i in [ii for ii in range(self.N) if ii not in P_t]:
                for l, option in enumerate(self.workers[i].options):
                    # Compute UCB quality difference (Equation 12)
                    ucb_diff = self.compute_ucb_quality(P_t | {i: option}) - self.compute_ucb_quality(P_t)
                    criterion = ucb_diff / option.cost
                    items.append((i, l, criterion))

            # Recruit the worker with the maximum ratio of marginal UCB to cost. (Line 7 & 8)
            if items:
                i_star, l_star, _ = max(items, key=lambda x: x[2])
                P_t[i_star] = self.workers[i_star].options[l_star]

        return P_t

    def run(self):
        """
        Run the UWR algorithm until the budget is exhausted.

        :return: Total utility achieved and the number of rounds conducted.
        """
        while True:
            self.tau += 1
            P_t = self.select_winners()

            # Terminate if the budget is exceeded.
            if sum(option.cost for option in P_t.values()) >= self.B:
                break

            self.update_profile(P_t)

        return self.U, self.tau
