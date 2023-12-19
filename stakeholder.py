"""
Author: DURUII
Date: 2023/12/18

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/arms.py
2. "II. SYSTEM MODEL & PROBLEM" of the paper
"""
import random

import numpy as np


class Task:
    def __init__(self, weight):
        # w, importance weight of the task
        self.w = weight


class SimpleOption:
    def __init__(self, tasks: list[int]):
        # indexes of the chosen tasks
        self.tasks = tasks


class Option(SimpleOption):
    def __init__(self, tasks: list[int], cost: float):
        super().__init__(tasks)

        # the total cost given such a set of tasks
        self.cost = cost


class Worker:
    # of options submitted to the platform
    L = 50

    def __init__(self, cost_parameter, expectation):
        # epsilon, heterogeneous, prior
        self.eps = cost_parameter

        # q, to formulate the i.i.d
        self.q = expectation

        # where every single item in the options to submit is (tasks, total cost)
        self.options = [Option() for l in range(Worker.L)]

        # sort by cost
        self.options.sort(key=lambda o: o.cost)

    def draw(self, gauss=True):
        """ empirical reward """
        if gauss:
            return random.gauss(0, 1)
        return random.uniform(0, 1)

    def cost(self, f, n):
        """ input the eps, f, |M^l|, output the cost which is proportional to f(|M^L|) """
        return self.eps * f(n)


class ExtendedWorker(Worker):
    # of options submitted to the platform
    L = 50
    eps_min = 0.1

    def __init__(self, cost_parameter, expectation):
        # epsilon, heterogeneous, prior
        super().__init__(cost_parameter, expectation)

        self.eps = cost_parameter

        # q, to formulate the i.i.d
        self.q = expectation

        # where every single item in the options to submit is (tasks, total cost)
        self.options = [SimpleOption() for l in range(Worker.L)]

    def draw(self, gauss=True):
        """ empirical reward """
        if gauss:
            return random.gauss(0, 1)
        return random.uniform(0, 1)

    def epsilon(self):
        pass
