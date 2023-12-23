"""
Author: DURUII
Date: 2023/12/23

Ref:
1. "Evaluation Methodology" in the paper
2. https://github.com/DURUII/Replica-EUWR/blob/main/processor.ipynb
"""
import random

import os

import numpy as np

from config import Config
import geopandas as gpd
import pandas as pd
from stakeholder import Worker, Task, ExtendedWorker, Option, SimpleOption
from dateutil import parser


class Generator:

    def __init__(self, num_tasks: int, num_workers: int, num_options: int, f,
                 extended=False, gauss_q=True, gauss_eps=True) -> None:
        self.N = num_workers
        self.M = num_tasks
        self.L = num_options
        self.f = f
        self.extended = extended
        self.gauss_q = gauss_q
        self.gauss_eps = gauss_eps

    def generate(self) -> (list[Task], list[Worker]):
        assert os.path.exists('./dataset/tasks.geojson')
        assert os.path.exists('./dataset/workers.csv')
        assert os.path.exists('./dataset/taxi_february.pkl') or os.path.exists('./dataset/taxi_february.txt')

        # select M locations as tasks
        tasks = gpd.read_file('./dataset/tasks.geojson')
        tasks = tasks.sample(n=self.M)

        tasks['buffer'] = tasks.geometry.buffer(200)
        tasks_active = tasks.set_geometry('buffer')
        tasks_active.drop(['DRIVER_ID', 'TIMESTAMP'], axis=1, inplace=True)

        while True:
            # preparation for counting the frequency
            if not os.path.exists('./dataset/taxi_february-random.geojson'):
                if not os.path.exists('./dataset/taxi_february.pkl'):
                    df = pd.read_csv('./dataset/taxi_february.txt',
                                     sep=';', header=None,
                                     names=['DRIVER_ID', 'TIMESTAMP', 'POSITION'])

                    df['TIMESTAMP'] = df['TIMESTAMP'].apply(lambda x: parser.parse(x))

                    df[['LATITUDE', 'LONGITUDE']] = df['POSITION'].str.extract(
                        r'POINT\(([^ ]+) ([^ ]+)\)')
                    df['LATITUDE'] = df['LATITUDE'].astype(float)
                    df['LONGITUDE'] = df['LONGITUDE'].astype(float)
                    df = df.drop('POSITION', axis=1)

                    df.to_pickle('./dataset/taxi_february.pkl')
                df = pd.read_pickle('./dataset/taxi_february.pkl')

                # speed up
                random.choice(df['TIMESTAMP'].dt.date.unique())
                date = random.choice(df['TIMESTAMP'].dt.date.unique())
                data = df[df['TIMESTAMP'].dt.date == date]
                gdf = gpd.GeoDataFrame(data.drop(['LONGITUDE', 'LATITUDE'], axis=1),
                                       geometry=gpd.points_from_xy(data['LONGITUDE'], data['LATITUDE']),
                                       crs='EPSG:4326').to_crs(epsg=3857)

                gdf.to_file('./dataset/taxi_february-random.geojson')
            gdf = gpd.read_file('./dataset/taxi_february-random.geojson')

            within_tasks_buffer = gpd.sjoin(gdf, tasks_active, how='inner', predicate='within')
            driver_counts_within = within_tasks_buffer['DRIVER_ID'].value_counts().reset_index()
            driver_counts_within.columns = ['DRIVER_ID', 'WITHIN']

            # select N drivers as workers
            driver = driver_counts_within.sample(n=self.N)
            q = driver['WITHIN'].values * 1.0
            q /= max(q)

            print(len(q), q)
            if len(q) == self.N:
                break

        # let w_j be uniform
        weights = [random.uniform(0, 1) for j in range(self.M)]
        T = [Task(weight=weights[j]) for j in range(self.M)]

        W = []
        for i in range(self.N):
            eps = random.uniform(Worker.eps_min, 1)
            D = lambda: min(random.uniform(0, 2 * q[i]), 1)
            if self.gauss_q:
                D = lambda: random.gauss(q[i], random.uniform(0, min(q[i] / 3, (1 - q[i] / 3))))

            if self.extended:
                options = []
                for l in range(self.L):
                    options.append(SimpleOption(random.sample(list(range(self.M)), k=random.randint(5, 15))))

                E = lambda: min(random.uniform(0, 2 * eps), 1)
                if self.gauss_eps:
                    E = lambda: random.gauss(eps, random.uniform(0, min(eps / 3, (1 - eps / 3))))

                w_i = ExtendedWorker(cost_parameter=eps,
                                     expectation=q[i],
                                     options=options,
                                     D=D, E=E)
            else:
                options = []
                for l in range(self.L):
                    t = random.sample(list(range(self.M)), k=random.randint(5, 15))
                    c = self.f(len(t)) * eps
                    options.append(Option(t, c))

                w_i = Worker(cost_parameter=random.uniform(0, 1),
                             expectation=q,
                             options=sorted(options, key=lambda o: o.cost),
                             D=D)

            W.append(w_i)
        return T, W


if __name__ == '__main__':
    config = Config
    tt, ww = Generator(config.M, config.N, config.L, config.f).generate()
