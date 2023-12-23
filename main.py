import os

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from config import Config
from emulator import Emulator
import pickle

plt.style.use(['science', 'grid'])
config = Config

# data preparation
if not os.path.exists('./runs.pkl'):
    data = []
    for X in ['N', 'M', 'rK', 'B']:
        for x in tqdm(eval(f'config.{X}_range'), desc=X):
            if X == 'N':
                name2res = Emulator(n_workers=x).simulate()
            elif X == 'M':
                name2res = Emulator(n_tasks=x).simulate()
            elif X == 'K':
                name2res = Emulator(r_selected=x).simulate()
            else:
                name2res = Emulator(budget=x).simulate()

            for key in name2res.keys():
                data.append([X, x, key, name2res[key][0], name2res[key][1]])

    df = pd.DataFrame(np.array(data), columns=['X', 'Val', 'Algorithm', 'Reward', 'Round'])

    with open('./runs.pkl', 'wb') as fout:
        pickle.dump(df, fout)

else:
    with open('./runs.pkl', 'rb') as fin:
        df = pickle.load(fin)

print(df)



