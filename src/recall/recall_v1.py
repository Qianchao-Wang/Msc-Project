import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.utils.data_utils import get_user_item_time_dict

import collections
import random
import math
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


class ItemCF(object):
    def __init__(self, args, behavior_dataset = None):
        self.args = args
        self.behavior = behavior_dataset
        self.user_item_time_dict = get_user_item_time_dict(self.behavior)

    # Item-based recall item2item
    def item_based_recommend(self, user_id, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
        """
            Recall based on item collaborative filtering
            :param user_id: user id
            :param i2i_sim: Dictionary, item similarity matrix
            :param sim_item_topk: Integer, Select the top k items that are most similar to the current item
            :param recall_item_num: Integer, The number of recalled items for the current user
            :param item_topk_click: List, The list of items with the most clicks used for the user's candidate items completion

            return: List of recalled items [(item1, score1), (item2, score2)...]
        """
        # Get user history clicked items
        user_hist_items = self.user_item_time_dict[user_id]
        user_hist_items_ = {item_id for item_id, _ in user_hist_items}

        item_rank = {}
        for loc, (i, click_time) in enumerate(user_hist_items):
            for j, sim_ij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items_:
                    continue

                # The weight of the position of the current item in the historical click item sequence
                loc_weight = (0.9 ** (len(user_hist_items) - loc))

                item_rank.setdefault(j, 0)
                item_rank[j] += loc_weight * sim_ij

        # if candidate items are less than recall_item_num, complete  with popular items
        if len(item_rank) < recall_item_num:
            for i, item in enumerate(item_topk_click):
                if item in item_rank.items():  # The filled item should not be in the original list
                    continue
                item_rank[item] = - i - 100
                if len(item_rank) == recall_item_num:
                    break

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

        return item_rank






