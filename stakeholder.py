"""
Author: DURUII
Date: 2023/12/18

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/arms.py
2. "II. SYSTEM MODEL & PROBLEM" of the paper
"""
import random


class SimpleOption:
    def __init__(self, tasks: list[int]):
        self.tasks = tasks

    def compute_cost(self, f, eps):
        return eps * f(len(self.tasks))


class Task:
    def __init__(self, weight):
        self.w = weight


class Worker:
    eps_min = 0.1

    def __init__(self, e: float, q: float, options: list[SimpleOption]):
        # eps (cost parameter) -> prior, fixed value
        self.mu_e = e
        self.mu_q = q
        self.sigma_q = random.uniform(0, min(q / 3, (1 - q) / 3))
        self.options = options

    def draw(self):
        return random.gauss(self.mu_q, self.sigma_q)


class ExtendedWorker(Worker):
    def __init__(self, e: float, q: float, options: list[SimpleOption]):
        super().__init__(e, q, options)
        self.sigma_e = random.uniform(0, min(e / 3, (1 - e) / 3))

    def epsilon(self):
        # eps (cost parameter) -> sampled from a distribution
        return random.gauss(self.mu_e, self.sigma_e)
