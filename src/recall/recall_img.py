import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import faiss
import sys
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_user_item_time_dict


class Imgrecall(object):
    def __init__(self, args, behavior_dataset=None, feature_dict=None, item_topk_click=None):
        self.args = args
        self.behavior = behavior_dataset
        self.user_item_time_dict = get_user_item_time_dict(self.behavior)
        self.item_topk_click = item_topk_click

        self.img_emb_df = pd.DataFrame()
        self.img_emb_df["feat_vec"] = feature_dict.values()
        self.img_emb_df['item_id'] = list(feature_dict.keys())
        self.img_emb_df["feat_vec"] = self.img_emb_df["feat_vec"].apply(lambda x: list(x[128:]))
        # Dictionary mapping of item index and item ID
        self.item_idx_2_rawid_dict = dict(zip(self.img_emb_df.index, self.img_emb_df['item_id']))
        self.item_rawid_2_idx_dict = dict(zip(self.img_emb_df["item_id"], self.img_emb_df.index))
        self.img_emb_np = np.ascontiguousarray(list(self.img_emb_df["feat_vec"].values), dtype=np.float32)
        self.user_list = self.behavior['user_id'].unique()

    def get_user_vec(self):
        users_vec = []
        user_idx_2_rawid_dict = {}
        user_rawid_2_idx_dict = {}
        for idx, user_id in enumerate(self.user_list):
            user_idx_2_rawid_dict[idx] = user_id
            user_rawid_2_idx_dict[user_id] = idx
            vec = []
            user_hist_items = self.user_item_time_dict[user_id]
            for loc, (i, click_time) in enumerate(user_hist_items):

                loc_weight = (0.7 ** (len(user_hist_items) - loc))
                item_txt_vec = self.img_emb_np[self.item_rawid_2_idx_dict[i]]
                if np.isnan(item_txt_vec)[0]:
                    vec.append([0] * 128)
                else:
                    vec.append(loc_weight * self.img_emb_np[self.item_rawid_2_idx_dict[i]])
            user_vec = np.mean(vec, axis=0)
            users_vec.append(user_vec)
        users_vec = np.array(users_vec).astype('float32')
        return users_vec, user_idx_2_rawid_dict, user_rawid_2_idx_dict

    def img_based_recommend(self):

        item_index = faiss.IndexFlatIP(self.img_emb_np.shape[1])
        item_index.add(self.img_emb_np)
        # The similarity query returns top k items and similarity to the vector at each index position
        users_vec, user_idx_2_rawid_dict, user_rawid_2_idx_dict = self.get_user_vec()
        sim, idx = item_index.search(users_vec, self.args.sim_item_topk + 1)
        user_recall_items_dict = defaultdict(dict)
        for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(users_vec)), sim, idx)):
            user_raw_id = user_idx_2_rawid_dict[target_idx]
            user_recall_items_dict[user_raw_id] = defaultdict(dict)
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                if rele_idx in self.item_idx_2_rawid_dict.keys():
                    rele_raw_id = self.item_idx_2_rawid_dict[rele_idx]
                    user_recall_items_dict[user_raw_id][rele_raw_id] = sim_value

            if len(user_recall_items_dict[user_raw_id]) < self.args.recall_item_num:
                for i, item in enumerate(self.item_topk_click):
                    if item in user_recall_items_dict[user_raw_id].items():  # The filled item should not be in the original list
                        continue
                    user_recall_items_dict[user_raw_id][item] = - i - 1
                    if len(user_recall_items_dict[user_raw_id]) == self.args.recall_item_num:
                        break
            user_recall_items_dict[user_raw_id] = sorted(user_recall_items_dict[user_raw_id].items(),
                                                         key=lambda x: x[1], reverse=True)[:self.args.recall_item_num]
        return user_recall_items_dict