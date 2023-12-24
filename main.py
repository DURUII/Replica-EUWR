import os

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from config import Config
from emulator import Emulator
import pickle

import scienceplots
plt.style.use(['science', 'grid'])
config = Config

for extended in [False, True]:
    config.extended = extended
    filepath = './extended-runs.pkl' if config.extended else './runs.pkl'

    # data preparation
    if not os.path.exists(filepath):
        data = []
        for X in ['N', 'M', 'rK', 'B']:
            for x in tqdm(eval(f'config.{X}_range'), desc=X):
                if X == 'N':
                    name2res = Emulator(n_workers=x, extended=config.extended).simulate()
                elif X == 'M':
                    name2res = Emulator(n_tasks=x, extended=config.extended).simulate()
                elif X == 'rK':
                    name2res = Emulator(r_selected=x, extended=config.extended).simulate()
                else:
                    name2res = Emulator(budget=x, extended=config.extended).simulate()

                data.extend(
                    [X, x, key, name2res[key][0], name2res[key][1]] for key in name2res.keys()
                )
        df = pd.DataFrame(np.array(data), columns=['X', 'Val', 'Algorithm', 'Reward', 'Round'])

        with open(filepath, 'wb') as fout:
            pickle.dump(df, fout)

    else:
        with open(filepath, 'rb') as fin:
            df = pickle.load(fin)

    # result visualization
    df['Val'] = df['Val'].astype(float)
    df['Reward'] = df['Reward'].astype(float)
    df['Round'] = df['Round'].astype(float)
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 12.5))

    df.drop_duplicates(subset=['X', 'Val', 'Algorithm'], keep='last', inplace=True)
    print(df)

    # line charts
    algos = [algo for algo in Emulator.algorithms if algo.startswith(
        'extended-')] if config.extended else [algo for algo in Emulator.algorithms if not algo.startswith('extended-')]

    for algo in algos:
        if config.extended:
            algo = algo[8:]
        data = df[(df.X == 'B') & (df.Algorithm == algo)]
        ax = axes[0, 1]
        ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
        ax.set_xlabel('Budget')
        ax.set_ylabel('Total rounds')

        data = df[(df.X == 'N') & (df.Algorithm == algo)]
        ax = axes[1, 1]
        ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
        ax.set_xlabel('Number of arms (N)')
        ax.set_ylabel('Total rounds')

        data = df[(df.X == 'M') & (df.Algorithm == algo)]
        ax = axes[2, 1]
        ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
        ax.set_xlabel('Number of tasks (M)')
        ax.set_ylabel('Total rounds')

        data = df[(df.X == 'rK') & (df.Algorithm == algo)]
        ax = axes[3, 1]
        ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
        ax.set_xlabel('Parameter (rK)')
        ax.set_ylabel('Total rounds')

    # bar plots
    n_algos = len(algos)

    for X, ax in zip(['B', 'N', 'M', 'rK'], [axes[0, 0], axes[1, 0], axes[2, 0], axes[3, 0]]):
        data = df[df.X == X].pivot(
            index='Val', columns='Algorithm', values='Reward')
        print(data.head())
        for i, algo in enumerate(algos):
            xpos = np.arange(len(data.index)) + \
                (i - n_algos / 2) * config.bar_width
            ax.bar(xpos, data[algo], width=config.bar_width,
                   **config.bar_styles[algo])

        ax.set_ylabel('Total rewards')
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index)

    axes[0, 0].set_xlabel('Budget')
    axes[1, 0].set_xlabel('Number of arms (N)')
    axes[2, 0].set_xlabel('Number of tasks (M)')
    axes[3, 0].set_xlabel('Parameter (rK)')
    axes[3, 0].set_xticklabels(config.rk_label)

    for ax in axes.flat:
        ax.legend()

    plt.savefig('fig-extended.jpg' if config.extended else 'fig.jpg', dpi=800)
