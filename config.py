"""
Author: DURUII
Date: 2023/12/23

Ref:
1. https://github.com/DURUII/Replica-AUCB/blob/main/config.py
2. "PERFORMANCE EVALUATION" in the paper
3. "TABLE I" in the paper
"""

import numpy as np


class Config:
    N = 50
    N_range = [50, 60, 70, 80, 90, 100]

    M = 300
    M_range = np.arange(1, 7) * 100

    rK = 1 / 3
    rK_range = [3 / 5, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6]
    rk_label = ["3/5", "1/2", "1/3", "1/4", "1/5", "1/6"]

    B = 1e4
    B_range = np.array([i for i in range(2, 11)]) * 1e3

    # NOTE: default value is not specified in the paper
    L = 2

    def f(x): 
        return x

    easy_generator = False
    extended = False

    line_styles = {
        'UWR': {'color': '#060506', 'marker': 's', 'label': 'Algo.1'},
        'EUWR': {'color': '#060506', 'marker': 's', 'label': 'Algo.2'},
        'opt': {'color': '#ed1e25', 'marker': 'o', 'label': 'Optimal'},
        '0.05-first': {'color': '#3753a4', 'marker': '^', 'label': '0.05-first'},
        '0.1-first': {'color': '#097f80', 'marker': 'v', 'label': '0.1-first'},
        'random': {'color': '#ba529e', 'marker': 'D', 'label': 'Random'},
    }

    # bar style
    bar_width = 0.15
    bar_styles = {
        'UWR': {'color': '#060506', 'hatch': '', 'label': 'Algo.1'},
        'EUWR': {'color': '#060506', 'marker': 's', 'label': 'Algo.2'},
        'opt': {'color': '#ed1e25', 'hatch': '|||||', 'label': 'Optimal'},
        '0.05-first': {'color': '#3753a4', 'hatch': '///////', 'label': '0.05-first'},
        '0.1-first': {'color': '#097f80', 'hatch': '\\\\\\\\\\\\', 'label': '0.1-first'},
        'random': {'color': '#ba529e', 'hatch': '-----', 'label': 'Random'},
    }
