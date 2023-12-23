"""
Author: DURUII
Created: 2023/12/18
Revised: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/arms.py
2. "II. SYSTEM MODEL & PROBLEM" of the paper
"""
import math
import random


class SimpleOption:
    c_max = 15

    def __init__(self, tasks: list[int]):
        self.tasks = tasks
        self.cost = 0.0

    def __str__(self):
        return f"SimpleOption(tasks={self.tasks})"

    def __repr__(self):
        return self.__str__()

    def update_cost(self, f, observed_eps):
        self.cost = observed_eps * f(len(self.tasks)) / SimpleOption.c_max
        # SimpleOption.c_max = max(self.cost, SimpleOption.c_max)
        # SimpleOption.c_min = min(self.cost, SimpleOption.c_min)
        return self


class Task:
    def __init__(self, weight: float):
        self.w = weight

    def __str__(self):
        return f"Task(weight={self.w})"

    def __repr__(self):
        return self.__str__()


class Worker:
    eps_min = 0.1

    def __init__(self, e: float, q: float, options: list[SimpleOption]):
        # in original problem, eps (cost parameter) -> prior, fixed value
        self.mu_e = e
        self.mu_q = q
        self.sigma_q = random.uniform(0, min(q / 3, (1 - q) / 3))
        self.options = options
        # if extended, eps (cost parameter) -> sampled from a distribution
        self.sigma_e = random.uniform(0, min(e / 3, (1 - e) / 3))

    def draw(self):
        """ universally used in algorithm 1 & 2 """
        return random.gauss(self.mu_q, self.sigma_q)

    def epsilon(self):
        """ only used in extended problem setting """
        return random.gauss(self.mu_e, self.sigma_e)

    def __str__(self):
        return f"Worker(e={self.mu_e}, q={self.mu_q}, options={self.options})"

    def __repr__(self):
        return self.__str__()
