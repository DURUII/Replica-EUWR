"""
Author: DURUII
Date: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/emulator.py
"""
from algorithms.euwr import EUWR
from algorithms.first import EpsilonFirst
from algorithms.opt import Opt
from algorithms.rand import Random
from algorithms.uwr import UWR
from generator import EasyGenerator, GeoGenerator
from config import Config as config
import time


class Emulator:
    algorithms = ['UWR', 'opt',  '0.05-first', '0.1-first', 'random',
                  'extended-EUWR', 'extended-opt',  'extended-0.05-first', 'extended-0.1-first', 'extended-random',
                  ]

    def __init__(self, n_tasks: int = config.M,
                 n_workers: int = config.N,
                 r_selected: float = config.rK,
                 n_options: int = config.L,
                 budget: float = config.B,
                 f=config.f,
                 easy_generator=config.easy_generator,
                 extended=config.extended):
        self.N = n_workers
        self.M = n_tasks
        self.B = budget
        self.K = int(r_selected * self.N)
        self.L = n_options
        self.f = f
        self.extended = extended

        if easy_generator:
            generator = EasyGenerator(n_tasks=self.M, n_workers=self.N, n_options=self.L, f=lambda x: x)
        else:
            generator = GeoGenerator(n_tasks=self.M, n_workers=self.N, n_options=self.L, f=lambda x: x)
        self.tasks, self.workers = generator.generate()
        self.name2sol = {}

    def build(self):
        for algo in Emulator.algorithms:
            if self.extended:
                if algo.startswith('extended'):
                    algo = algo[8:][:]
                    if algo == 'EUWR':
                        self.name2sol[algo] = EUWR(self.workers, self.tasks, self.K, self.B, self.f)
                    elif algo.endswith('-first'):
                        self.name2sol[algo] = EpsilonFirst(self.workers, self.tasks, self.K, self.B, self.f,
                                                           float(algo[:-6]), extended=True)
                    elif algo == 'random':
                        self.name2sol[algo] = Random(self.workers, self.tasks, self.K, self.B, self.f, extended=True)
                    elif algo == 'opt':
                        self.name2sol[algo] = Opt(self.workers, self.tasks, self.K, self.B, self.f, extended=True)

            elif not algo.startswith('extended'):
                if algo == 'UWR':
                    self.name2sol[algo] = UWR(self.workers, self.tasks, self.K, self.B, self.f)
                elif algo.endswith('-first'):
                    self.name2sol[algo] = EpsilonFirst(self.workers, self.tasks, self.K, self.B, self.f,
                                                       float(algo[:-6]))
                elif algo == 'random':
                    self.name2sol[algo] = Random(self.workers, self.tasks, self.K, self.B, self.f)
                elif algo == 'opt':
                    self.name2sol[algo] = Opt(self.workers, self.tasks, self.K, self.B, self.f)

    def simulate(self):
        self.build()
        name2res = {name: None for name in self.name2sol.keys()}
        for name in name2res:
            tic = time.perf_counter()
            # instance of an algorithm
            solver = self.name2sol[name]
            solver.initialize()
            name2res[name] = solver.run()
            toc = time.perf_counter()
            print(f'algo={name}, res={name2res[name]}, time={toc-tic}')
        return name2res
