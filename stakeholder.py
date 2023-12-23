"""
Author: DURUII
Date: 2023/12/18

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/arms.py
2. "II. SYSTEM MODEL & PROBLEM" of the paper
"""


class SimpleOption:
    def __init__(self, tasks: list[int]):
        # indexes of the chosen tasks
        self.tasks = tasks

    def compute_cost(self, f, eps):
        return eps * f(len(self.tasks))


class Option(SimpleOption):
    def __init__(self, tasks: list[int], cost: float):
        super().__init__(tasks)

        # the total cost given such a set of tasks
        self.cost = cost


class Task:
    def __init__(self, weight):
        self.w = weight


class Worker:
    eps_min = 0.1

    def __init__(self, cost_parameter: float, expectation: float, options: list[Option], D):
        self.eps = cost_parameter
        self.q = expectation
        self.options = options
        self.D = D

    def draw(self):
        return self.D()


class ExtendedWorker(Worker):
    def __init__(self, cost_parameter: float, expectation: float, options: list[SimpleOption], D, E):
        super().__init__(cost_parameter, expectation, [], D)
        self.options: list[SimpleOption] = options
        self.E = E

    def epsilon(self):
        return self.E()
