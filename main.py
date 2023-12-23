from algorithms.euwr import EUWR
from algorithms.uwr import UWR
from config import Config
from generator import Generator

config = Config
tasks, workers = Generator(config.M, config.N, config.L, config.f,
                           extended=True).generate()
print(len(tasks), len(workers))
solver = EUWR(workers, tasks, config.rK, config.B, config.f)
solver.initialize()
print(solver.run())
