import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import collections
import random
import math
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import sys
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_user_item_time_dict



class ItemCF(object):
    def __init__(self, args, behavior_dataset=None, i2i_sim=None, item_topk_click=None):
        """
        Initialization class
        :param args: some Hyper parameters
        :param behavior_dataset: Dataframe that records the user's historical behavior
        :param i2i_sim: Dictionary, item similarity matrix
        """
        self.args = args
        self.behavior = behavior_dataset
        self.user_item_time_dict = get_user_item_time_dict(self.behavior)
        # Get the list of items with the most clicks used for the user's candidate items completion
        self.item_topk_click = item_topk_click
        self.i2i_sim = i2i_sim

    # Item-based recall item2item
    def item_based_recommend(self, user_id):
        """
            Recall based on item collaborative filtering
            :param user_id: user id
            return: Recommended candidate set [(item1, score1), (item2, score2)...]
        """
        # Get user history clicked items
        user_hist_items = self.user_item_time_dict[user_id]
        user_hist_items_ = {item_id for item_id, _ in user_hist_items}

        item_rank = {}
        for loc, (i, click_time) in enumerate(user_hist_items):
            for j, sim_ij in sorted(self.i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:self.args.sim_item_topk]:
                if j in user_hist_items_:
                    continue

                # The weight of the position of the current item in the historical click item sequence
                loc_weight = (0.9 ** (len(user_hist_items) - loc))

                item_rank.setdefault(j, 0)
                # Calculate similarity with position weight
                item_rank[j] += loc_weight * sim_ij

        # if candidate items are less than recall_item_num, complete  with popular items
        if len(item_rank) < self.args.recall_item_num:
            for i, item in enumerate(self.item_topk_click):
                if item in item_rank.items():  # The filled item should not be in the original list
                    continue
                item_rank[item] = - i - 100
                if len(item_rank) == self.args.recall_item_num:
                    break

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:self.args.recall_item_num]

        return item_rank






