import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle
import math
import logging
import sys
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_item_user_time_dict
from src.utils.data_utils import get_user_activate_degree_dict
from src.utils.data_utils import get_hist_and_last_click
from src.data_process.load_data import get_all_click_data


def usercf_sim(behavior, save_path):
    """
    Calculate user similarity matrix
    :param behavior: Dataframe that records the user's historical behavior
    :param save_path:
    :return: Dictionary, similarity matrix of users
    """
    item_user_time_dict = get_item_user_time_dict(behavior)
    user_activate_degree_dict = get_user_activate_degree_dict(behavior)

    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                u2u_sim[u][v] += 1
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] + user_cnt[v])

    # Save the obtained similarity matrix to the local
    pickle.dump(u2u_sim_, open(save_path + 'usercf_u2u_sim_v1.pkl', 'wb'))

    return u2u_sim_


if __name__ == '__main__':
    print("--------load data----------------------")
    all_click, click_test = get_all_click_data("online")
    save_path = "output/online/similarity/"
    print("---------------calculate sim------------")
    usercf_sim(all_click, save_path)