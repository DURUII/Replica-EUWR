"""
Author: DURUII
Created: 2023/12/23

Ref:
1. "Evaluation Methodology" in the paper
2. https://github.com/DURUII/Replica-EUWR/blob/main/processor.ipynb
"""
import random

import os

import numpy as np
from typing import Iterable

from config import Config
import geopandas as gpd
import pandas as pd
from stakeholder import Worker, Task, SimpleOption
from dateutil import parser
from rich import print


class EasyGenerator:
    def __init__(self, n_tasks: int, n_workers: int, n_options: int, f):
        self.N = n_workers
        self.M = n_tasks
        self.L = n_options
        self.f = f

    def generate_tasks(self, weights: list[float]) -> list[Task]:
        if not weights:
            weights = np.diff(sorted([random.uniform(0, 1) for j in range(self.M + 1)]))

        return [Task(weights[j]) for j in range(self.M)]

    def generate_workers(self, q, e) -> list[Worker]:
        workers = []
        if not q:
            q = [random.uniform(0.1, 1) for i in range(self.N)]
        if not e:
            e = [random.uniform(0.1, 1) for i in range(self.N)]

        workers = [
            Worker(e[i], q[i],
                   [SimpleOption(random.sample(range(self.M), k=random.randint(5, 15))) for l in range(self.L)])
            for i in range(self.N)
        ]

        return workers

    def generate(self) -> (list[Task], list[Worker]):
        return self.generate_tasks([]), self.generate_workers([], [])


class GeoGenerator(EasyGenerator):
    def __init__(self, n_tasks: int, n_workers: int, n_options: int, f) -> None:
        super().__init__(n_tasks, n_workers, n_options, f)

    def generate(self) -> (list[Task], list[Worker]):
        assert os.path.exists('./dataset/tasks.geojson')
        assert os.path.exists('./dataset/workers.csv')
        assert os.path.exists('./dataset/taxi_february.pkl') or os.path.exists('./dataset/taxi_february.txt')

        # select M locations
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

            # select N drivers
            within_tasks_buffer = gpd.sjoin(gdf, tasks_active, how='inner', predicate='within')
            driver_counts_within = within_tasks_buffer['DRIVER_ID'].value_counts().reset_index()
            driver_counts_within.columns = ['DRIVER_ID', 'WITHIN']
            driver = driver_counts_within.sample(n=self.N)
            q = driver['WITHIN'].values * 1.0
            q /= max(q)
            if len(q) == self.N:
                break

        return self.generate_tasks([]), self.generate_workers(q=list(q), e=[])


if __name__ == '__main__':
    config = Config
    tt, ww = EasyGenerator(config.M, config.N, config.L, config.f).generate()
