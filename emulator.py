"""
Author: DURUII
Date: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/emulator.py
"""

from algorithms.uwr import UWR
from generator import EasyGenerator, GeoGenerator


class Emulator:
    algorithms = ['UWR']

    def __init__(self, n_tasks: int = 300,
                 n_workers: int = 50,
                 r_selected: float = 1 / 3,
                 n_options: int = 5,
                 budget: float = 1e4,
                 f=lambda x: x,
                 easy_generator=True):
        self.N = n_workers
        self.M = n_tasks
        self.B = budget
        self.K = int(r_selected * self.N)
        self.L = n_options
        self.f = f

        if easy_generator:
            generator = EasyGenerator(n_tasks=self.M, n_workers=self.N, n_options=self.L, f=lambda x: x)
        else:
            generator = GeoGenerator(n_tasks=self.M, n_workers=self.N, n_options=self.L, f=lambda x: x)
        self.tasks, self.workers = generator.generate()
        self.name2sol = {}

    def build(self):
        for algo in Emulator.algorithms:
            if algo == 'UWR':
                self.name2sol[algo] = UWR(self.workers, self.tasks, self.K, self.B)

    def simulate(self):
        self.build()
        name2res = {name: None for name in self.name2sol.keys()}
        for name in name2res.keys():
            # instance of an algorithm
            solver = self.name2sol[name]
            solver.initialize()
            name2res[name] = solver.run()
            print(name2res)
        return name2res
