import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle
import math
import logging
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
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += 1
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] + item_cnt[j])

    # Save the obtained similarity matrix to the local
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim_v1.pkl', 'wb'))

    return i2i_sim_


if __name__ == '__main__':
    print("--------load data----------------------")
    all_click, click_test = get_all_click_data("online")
    save_path = "output/online/similarity/"
    print("---------------calculate sim------------")
    itemcf_sim(all_click, save_path)