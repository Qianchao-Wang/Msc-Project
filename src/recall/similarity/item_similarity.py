import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle
import math
import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_user_item_time_dict
from src.utils.data_utils import get_hist_and_last_click
from src.data_process.load_data import get_all_click_data


def itemcf_sim(behavior, save_path):
    """
    calculate similarity matrix between items
    :param behavior:  Dataframe that records the user's historical behavior
    :param save_path:
    :return: Dictionary, similarity matrix of items
    """
    user_item_time_dict = get_user_item_time_dict(behavior)

    # Calculate the similarity of items
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # Time factor can be considered when optimizing item-based collaborative filtering
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                # Consider the location feature of items
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # Position weight, the parameters can be adjusted
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # Click time weight, the parameters can be adjusted
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                i2i_sim[i].setdefault(j, 0)
                # Consider the weights of multiple factors to calculate the similarity between the items
                i2i_sim[i][j] += loc_weight * click_time_weight / math.log(len(item_time_list) + 1)


    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # Save the obtained similarity matrix to the local
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, required=True, type=str, help="online or offline")
    args = parser.parse_args()
    print("--------load data----------------------")
    all_click, click_test = get_all_click_data(args.mode)
    if args.mode == "online":
        user_hist_click = all_click
    elif args.mode == "offline":
        user_hist_click, user_last_click = get_hist_and_last_click(all_click)
    print("---------------calculate sim------------")
    save_path = "output/{}/similarity/".format(args.mode)
    itemcf_sim(user_hist_click, save_path)
