import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_user_item_time_dict
from src.data_process.load_data import get_all_click_data


def get_item_similarity_feature(i2i_sim_dict, label_df, behavior_df, task='item_sim'):
    """

    :param i2i_sim_dict:
    :param label_df:
    :param behavior_df:
    :return:
    """
    max_values, mean_values, var_values = [], [], []
    user_item_time_dict = get_user_item_time_dict(behavior_df)
    for index, data in tqdm(label_df.iterrows()):
        user_id = int(data.user_id)
        sim_item = int(data.sim_item)
        user_hist_items = user_item_time_dict[user_id]
        similarity = []
        for (item, click_time) in user_hist_items:
            if item in i2i_sim_dict.keys() and sim_item in i2i_sim_dict[item].keys():
                similarity.append(i2i_sim_dict[item][sim_item])  # ?
            else:
                similarity.append(0)

        max_values.append(np.max(similarity))
        mean_values.append(np.mean(similarity))
        var_values.append(np.var(similarity))

    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["max_{}".format(task)] = max_values
    feature_df["mean_{}".format(task)] = mean_values
    feature_df["var_{}".format(task)] = var_values

    return feature_df


def get_nearest_similarity_feature(i2i_sim_dict, label_df, behavior_df, k=3, task='item_sim'):
    """

    :param i2i_sim_dict:
    :param label_df:
    :param behavior_df:
    :param k:
    :return:
    """
    total_sim = []
    user_item_time_dict = get_user_item_time_dict(behavior_df)
    for index, data in tqdm(label_df.iterrows()):
        user_id = int(data.user_id)
        sim_item = int(data.sim_item)
        user_hist_items = user_item_time_dict[user_id]
        sim = []
        if len(user_hist_items) >= k:
            for (item, click_time) in user_hist_items[-k:]:
                if item in i2i_sim_dict.keys() and sim_item in i2i_sim_dict[item].keys():
                    sim.append(i2i_sim_dict[item][sim_item])
                else:
                    sim.append(0)
        else:
            for i in range(k - len(user_hist_items)):
                sim.append(0)
            for (item, click_time) in user_hist_items:
                if item in i2i_sim_dict.keys() and sim_item in i2i_sim_dict[item].keys():
                    sim.append(i2i_sim_dict[item][sim_item])
                else:
                    sim.append(0)
        total_sim.append(sim)

    sim_np = np.array(total_sim)
    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["{}_1".format(task)] = sim_np[:, 0]
    feature_df["{}_2".format(task)] = sim_np[:, 1]
    feature_df["{}_3".format(task)] = sim_np[:, 2]

    return feature_df


if __name__ == "__main__":
    sim_dir = "output/offline/similarity/"
    i2i_sim_dict = pickle.load(open(os.path.join(sim_dir, 'itemcf_i2i_sim.pkl'), 'rb'))
    img_sim_dict = pickle.load(open(os.path.join(sim_dir, 'img_i2i_sim.pkl'), 'rb'))
    txt_sim_dict = pickle.load(open(os.path.join(sim_dir, 'txt_i2i_sim.pkl'), 'rb'))
    w2v_sim_dict = pickle.load(open(os.path.join(sim_dir, 'w2v_i2i_sim.pkl'), 'rb'))

    trn_label_df = pd.read_csv("Datasets/rank/trn_user_item_label_df.csv", sep=',', encoding='utf-8')
    tst_label_df = pd.read_csv("Datasets/rank/tst_user_item_label_df.csv", sep=',', encoding='utf-8')

    all_click, test_click = get_all_click_data("online")

    trn_item_sim_feat_df = get_item_similarity_feature(i2i_sim_dict, trn_label_df, all_click, "item_sim")
    trn_img_sim_feat_df = get_item_similarity_feature(img_sim_dict, trn_label_df, all_click, "img_sim")
    trn_txt_sim_feat_df = get_item_similarity_feature(txt_sim_dict, trn_label_df, all_click, "txt_sim")
    trn_w2v_sim_feat_df = get_item_similarity_feature(w2v_sim_dict, trn_label_df, all_click, "w2v_sim")

    trn_label_df = trn_label_df.merge(trn_item_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_img_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_txt_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_w2v_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df.to_csv("Datasets/rank/trn_user_item_label_feat_df.csv")

    """
    tst_item_sim_feat_df = get_item_similarity_feature(i2i_sim_dict, tst_label_df, all_click, "item_sim")
    tst_img_sim_feat_df = get_item_similarity_feature(img_sim_dict, tst_label_df, all_click, "img_sim")
    tst_txt_sim_feat_df = get_item_similarity_feature(txt_sim_dict, tst_label_df, all_click, "txt_sim")
    tst_w2v_sim_feat_df = get_item_similarity_feature(w2v_sim_dict, tst_label_df, all_click, "w2v_sim")
    
    tst_label_df = tst_label_df.merge(tst_item_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_img_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_txt_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_w2v_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df.to_csv("Datasets/rank/tst_user_item_label_feat_df.csv")
    """




