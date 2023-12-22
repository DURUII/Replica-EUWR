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

    K = 1 / 3
    K_range = [3 / 5, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6]

    B = 1e4
    B_range = np.array([i for i in range(2, 11)]) * 1e3

    # line_styles = {
    #     'AUCB': {'color': '#060506', 'marker': 's', 'label': 'AUCB'},
    #     'optimal': {'color': '#ed1e25', 'marker': 'o', 'label': 'optimal'},
    #     'separated': {'color': '#3753a4', 'marker': '^', 'label': 'separated'},
    #     '0.1-first': {'color': '#097f80', 'marker': 'v', 'label': '0.1-first'},
    #     '0.5-first': {'color': '#ba529e', 'marker': '<', 'label': '0.5-first'},
    # }
    #
    # # bar style
    # bar_width = 0.15
    # bar_styles = {
    #     'AUCB': {'color': '#060506', 'label': 'AUCB', 'hatch': ''},
    #     'optimal': {'color': '#ed1e25', 'label': 'optimal', 'hatch': '||||'},
    #     'separated': {'color': '#3753a4', 'label': 'separated', 'hatch': '/////'},
    #     '0.1-first': {'color': '#097f80', 'label': '0.1-first', 'hatch': '\\\\\\\\\\'},
    #     '0.5-first': {'color': '#ba529e', 'label': '0.5-first', 'hatch': '---'},
    # }
