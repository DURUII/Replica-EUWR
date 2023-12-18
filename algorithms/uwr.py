"""
Author: DURUII
Date: 2023/12/18

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/algorithms/aucb.py
2. pseudocode and the corresponding equation definition in the paper
"""
import numpy as np

from algorithms.base import BaseAlgorithm
from stakeholder import Worker, Task, Option
from collections import defaultdict


class UWR(BaseAlgorithm):
    def __init__(self, workers: list[Worker], tasks: list[Task], n_selected: int, n_options: int, budget: float):
        super().__init__(workers, tasks, n_selected, n_options, budget)

        # importance weight of a given task j
        self.w = np.array([self.tasks[j].w for j in range(self.M)])

        # storage, you can use P[i][l] to access p_i^l = (tasks, cost)
        self.P = {i: w_i.options for i, w_i in enumerate(self.workers)}

        # n_i(t), count for how many times each arm/worker {i} has been learned
        self.n = np.zeros(self.N)

        # \bar{q}_i, record the average empirical quality value (reward) of arm/worker {i}
        self.q_bar = np.zeros(self.N)

    def initialize(self):
        """ recruit all workers at the very beginning. """

        # STEP ONE: SELECT
        # you can use temp P_t[i] to access the selected option p_i^0 = (tasks, cost)
        P_t: dict[int, Option] = {i: w_i.options[0] for i, w_i in enumerate(self.workers)}

        # STEP TWO: UPDATE
        # q[i][j] means observed reward of task j completed by worker i
        q: dict[int, dict[int, int]] = defaultdict(dict)

        # u_tt[j] means final completion quality of task j
        u_tt = np.array([.0 for j in range(self.M)])

        # u_ww[i] means final completion quality of worker i
        u_ww = np.array([.0 for i in range(self.N)])

        for i in P_t.keys():
            for j in P_t[i].tasks:
                q[i][j] = self.workers[i].draw()
                # Equation 1
                u_tt[j] = max(q[i][j], u_tt[j])
                # Prepare for Equation 9
                u_ww[i] += q[i][j]

        # Equation 2
        self.U += np.sum(self.w * u_tt)

        # Line 2
        self.B -= np.sum(np.array([P_t[i].cost for i in P_t.keys()]))

        # | M_i^t |
        cardinality = np.array([len(P_t.get(i, Option(tasks=[], cost=0))) for i in range(self.N)])
        mask = np.zeros(self.N)
        mask[P_t.keys()] = 1

        # Equation 9
        self.q_bar = self.q_bar * (1 - mask) + (self.q_bar * self.n + u_ww) / (self.n + cardinality) * mask
        self.n += cardinality

    def run(self):
        # loop (Line 3)
        while True:
            # clear all (Line 4)
            self.tau += 1
            P_t: dict[int, Option] = {}

            # STEP ONE: SELECT
            # K workers with its option (Line 5)
            n_sum = np.sum(self.n)
            # Equation 10
            q_hat = self.q_bar + np.sqrt((self.K + 1) * np.log(n_sum) / self.n)

            def u(P_tt: dict[int, Option]):
                # \hat{q}_i(t-1) \cdot \mathbb{I} \{{j \in \mathcal{M} _i^l, p_i^l \in \mathcal{P}^t} \} (Equation 11)
                vv = np.zeros(self.M)
                for ii in P_tt.keys():
                    for jj in P_tt[i].tasks:
                        vv[jj] = max(q_hat[i], vv[jj])

                return np.sum(self.w * vv)

            while len(P_t) < self.K:
                criterion = []
                # Line 7
                for i in range(self.N):
                    if i not in P_t:
                        P_t_prime = P_t[:]
                        for l in range(self.workers[i].options):
                            option = self.workers[i].options[l]
                            P_t_prime[i] = option
                            criterion.append((i, l, (u(P_t_prime) - u(P_t)) / option.cost))

                # Line 7
                i, l, _ = max(criterion, key=lambda x: x[2])

                # Line 8
                P_t[i] = self.workers[i].options[l]

            # Line 9 & 10
            if np.sum(np.array([P_t[i].cost for i in P_t.keys()])) >= self.B:
                break

            # STEP TWO: UPDATE
            # q[i][j] means observed reward of task j completed by worker i
            q: dict[int, dict[int, int]] = defaultdict(dict)

            # u_tt[j] means final completion quality of task j
            u_tt = np.array([.0 for j in range(self.M)])
            # u_ww[i] means final completion quality of worker i
            u_ww = np.array([.0 for i in range(self.N)])
            for i in P_t.keys():
                for j in P_t[i].tasks:
                    q[i][j] = self.workers[i].draw()
                    # Equation 1
                    u_tt[j] = max(q[i][j], u_tt[j])
                    # Prepare for Equation 9
                    u_ww[i] += q[i][j]

            # Equation 2
            self.U += np.sum(self.w * u_tt)

            # Line 2
            self.B -= np.sum(np.array([P_t[i].cost for i in P_t.keys()]))

            # | M_i^t |
            cardinality = np.array([len(P_t.get(i, Option(tasks=[], cost=0))) for i in range(self.N)])
            mask = np.zeros(self.N)
            mask[P_t.keys()] = 1

            # Equation 9
            self.q_bar = self.q_bar * (1 - mask) + (self.q_bar * self.n + u_ww) / (self.n + cardinality) * mask
            self.n += cardinality

        return self.U, self.tau
