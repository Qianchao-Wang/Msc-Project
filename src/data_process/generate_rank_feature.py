import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_user_item_time_dict
from src.data_process.load_data import get_all_click_data
from src.utils.data_utils import get_cos_sim
from src.utils.data_utils import get_user_activate_degree_dict
from src.utils.data_utils import get_item_popular_degree_dict
from src.utils.data_utils import get_hist_and_last_click


def get_item_similarity_feature(i2i_sim_dict, label_df, behavior_df):
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
                similarity.append(0.0)

        max_values.append(np.max(similarity))
        mean_values.append(np.mean(similarity))
        var_values.append(np.var(similarity))
    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["max_i2i_sim"] = max_values
    feature_df["mean_i2i_sim"] = mean_values
    feature_df["var_i2i_sim"] = var_values
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    feature_df["max_i2i_sim"] = feature_df[["max_i2i_sim"]].apply(max_min_scaler)
    feature_df["mean_i2i_sim"] = feature_df[["mean_i2i_sim"]].apply(max_min_scaler)
    feature_df["var_i2i_sim"] = feature_df[["var_i2i_sim"]].apply(max_min_scaler)

    return feature_df


def get_w2v_similarity_feature(w2v_emd_dict, label_df, behavior_df):
    max_values, mean_values, var_values = [], [], []
    user_item_time_dict = get_user_item_time_dict(behavior_df)
    for index, data in tqdm(label_df.iterrows()):
        user_id = int(data.user_id)
        sim_item = int(data.sim_item)
        user_hist_items = user_item_time_dict[user_id]
        cos_sim = []
        for (item, click_time) in user_hist_items:
            if str(item) in w2v_emd_dict.keys() and str(sim_item) in w2v_emd_dict.keys():
                cos_sim.append(get_cos_sim(w2v_emd_dict[str(item)], w2v_emd_dict[str(sim_item)]))
            else:
                cos_sim.append(0.0)

        max_values.append(np.max(cos_sim))
        mean_values.append(np.mean(cos_sim))
        var_values.append(np.var(cos_sim))
    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["max_w2v_sim"] = max_values
    feature_df["mean_w2v_sim"] = mean_values
    feature_df["var_w2v_sim"] = var_values
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    feature_df["max_w2v_sim"] = feature_df[["max_w2v_sim"]].apply(max_min_scaler)
    feature_df["mean_w2v_sim"] = feature_df[["mean_w2v_sim"]].apply(max_min_scaler)
    feature_df["var_w2v_sim"] = feature_df[["var_w2v_sim"]].apply(max_min_scaler)

    return feature_df


def get_content_similarity_feature(emd_dict, label_df, behavior_df):
    max_values, mean_values, var_values = [], [], []
    user_item_time_dict = get_user_item_time_dict(behavior_df)
    for index, data in tqdm(label_df.iterrows()):
        user_id = int(data.user_id)
        sim_item = str(int(data.sim_item))
        user_hist_items = user_item_time_dict[user_id]
        cos_sim = []
        for (item, click_time) in user_hist_items:
            if str(item) in emd_dict.keys() and sim_item in emd_dict.keys():
                if np.isnan(emd_dict[str(item)])[0] or np.isnan(emd_dict[sim_item])[0]:
                    cos_sim.append(0.0)
                else:
                    cos_sim.append(get_cos_sim(emd_dict[str(item)], emd_dict[sim_item]))
            else:
                cos_sim.append(0.0)
        max_values.append(np.max(cos_sim))
        mean_values.append(np.mean(cos_sim))
        var_values.append(np.var(cos_sim))
    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["max_content_sim"] = max_values
    feature_df["mean_content_sim"] = mean_values
    feature_df["var_content_sim"] = var_values
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    feature_df["max_content_sim"] = feature_df[["max_content_sim"]].apply(max_min_scaler)
    feature_df["mean_content_sim"] = feature_df[["mean_content_sim"]].apply(max_min_scaler)
    feature_df["var_content_sim"] = feature_df[["var_content_sim"]].apply(max_min_scaler)

    return feature_df


def get_nearest_similarity_feature(i2i_sim_dict, w2v_emd_dict, txt_emd_dict, label_df, behavior_df, k=3):
    """

    :param i2i_sim_dict:
    :param label_df:
    :param behavior_df:
    :param k:
    :return:
    """
    total_i2i_sim = []
    total_w2v_sim = []
    total_txt_img_sim = []
    user_item_time_dict = get_user_item_time_dict(behavior_df)
    for index, data in tqdm(label_df.iterrows()):
        user_id = int(data.user_id)
        sim_item = int(data.sim_item)
        user_hist_items = user_item_time_dict[user_id]
        i2i_sim = []
        w2v_sim = []
        txt_img_sim = []
        if len(user_hist_items) >= k:
            for (item, click_time) in user_hist_items[-k:]:
                if item in i2i_sim_dict.keys() and sim_item in i2i_sim_dict[item].keys():
                    i2i_sim.append(i2i_sim_dict[item][sim_item])
                else:
                    i2i_sim.append(0.0)
                if str(item) in w2v_emd_dict.keys() and str(sim_item) in w2v_emd_dict.keys():
                    w2v_sim.append(get_cos_sim(w2v_emd_dict[str(item)], w2v_emd_dict[str(sim_item)]))
                else:
                    w2v_sim.append(0.0)
                if str(item) in txt_emd_dict.keys() and str(sim_item) in txt_emd_dict.keys():
                    txt_img_sim.append(get_cos_sim(txt_emd_dict[str(item)], txt_emd_dict[str(sim_item)]))
                else:
                    txt_img_sim.append(0.0)

        else:
            for i in range(k - len(user_hist_items)):
                i2i_sim.append(0.0)
                w2v_sim.append(0.0)
                txt_img_sim.append(0.0)
            for (item, click_time) in user_hist_items:
                if item in i2i_sim_dict.keys() and sim_item in i2i_sim_dict[item].keys():
                    i2i_sim.append(i2i_sim_dict[item][sim_item])
                else:
                    i2i_sim.append(0.0)
                if str(item) in w2v_emd_dict.keys() and str(sim_item) in w2v_emd_dict.keys():
                    w2v_sim.append(get_cos_sim(w2v_emd_dict[str(item)], w2v_emd_dict[str(sim_item)]))
                else:
                    w2v_sim.append(0.0)
                if str(item) in txt_emd_dict.keys() and str(sim_item) in txt_emd_dict.keys():
                    txt_img_sim.append(get_cos_sim(txt_emd_dict[str(item)], txt_emd_dict[str(sim_item)]))
                else:
                    txt_img_sim.append(0.0)
        total_i2i_sim.append(i2i_sim)
        total_w2v_sim.append(w2v_sim)
        total_txt_img_sim.append(txt_img_sim)

    i2i_sim_np = np.array(total_i2i_sim)
    w2v_sim_np = np.array(total_w2v_sim)
    txt_img_sim_np = np.array(total_txt_img_sim)
    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["item_sim_1"] = i2i_sim_np[:, 0]
    feature_df["item_sim_2"] = i2i_sim_np[:, 1]
    feature_df["item_sim_3"] = i2i_sim_np[:, 2]
    feature_df["w2v_sim_1"] = w2v_sim_np[:, 0]
    feature_df["w2v_sim_2"] = w2v_sim_np[:, 1]
    feature_df["w2v_sim_3"] = w2v_sim_np[:, 2]
    feature_df["content_sim_1"] = txt_img_sim_np[:, 0]
    feature_df["content_sim_2"] = txt_img_sim_np[:, 1]
    feature_df["content_sim_3"] = txt_img_sim_np[:, 2]

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    feature_df["item_sim_1"] = feature_df[["item_sim_1"]].apply(max_min_scaler)
    feature_df["item_sim_2"] = feature_df[["item_sim_2"]].apply(max_min_scaler)
    feature_df["item_sim_3"] = feature_df[["item_sim_3"]].apply(max_min_scaler)
    feature_df["w2v_sim_1"] = feature_df[["w2v_sim_1"]].apply(max_min_scaler)
    feature_df["w2v_sim_2"] = feature_df[["w2v_sim_2"]].apply(max_min_scaler)
    feature_df["w2v_sim_3"] = feature_df[["w2v_sim_3"]].apply(max_min_scaler)
    feature_df["content_sim_1"] = feature_df[["content_sim_1"]].apply(max_min_scaler)
    feature_df["content_sim_2"] = feature_df[["content_sim_2"]].apply(max_min_scaler)
    feature_df["content_sim_3"] = feature_df[["content_sim_3"]].apply(max_min_scaler)

    return feature_df


def get_users_activate_feature(label_df, behavior_df):
    activate_degree_dict = get_user_activate_degree_dict(behavior_df)
    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["user_activate"] = label_df.user_id.values
    feature_df["user_activate"] = feature_df["user_activate"].apply(lambda x: activate_degree_dict[x])

    return feature_df


def get_items_popular_feature(label_df, behavior_df):
    popular_degree_dict = get_item_popular_degree_dict(behavior_df)
    feature_df = pd.DataFrame()
    feature_df["user_id"] = label_df.user_id.values
    feature_df["sim_item"] = label_df.sim_item.values
    feature_df["item_popular"] = label_df.sim_item.values
    feature_df["item_popular"] = feature_df["item_popular"].apply(lambda x: popular_degree_dict[x])

    return feature_df


if __name__ == "__main__":
    offline_sim_dir = "output/offline/similarity/"
    i2i_sim_dict = pickle.load(open(os.path.join(offline_sim_dir, 'itemcf_i2i_sim_v2.pkl'), 'rb'))
    content_emd_dict = pickle.load(open(os.path.join("Datasets/offline", 'item_content_vec_dict.pkl'), 'rb'))
    w2v_emd_dict = pickle.load(open(os.path.join("Datasets/", 'item_w2v_emb.pkl'), 'rb'))

    trn_label_df = pd.read_csv("Datasets/rank/trn_user_item_label_df.csv", sep=',', encoding='utf-8')
    tst_label_df = pd.read_csv("Datasets/rank/tst_user_item_label_df.csv", sep=',', encoding='utf-8')
    trn_click, test_click = get_all_click_data("offline")
    trn_hist_click, trn_last_click = get_hist_and_last_click(trn_click)

    trn_item_sim_feat_df = get_item_similarity_feature(i2i_sim_dict, trn_label_df, trn_hist_click)
    trn_content_sim_feat_df = get_content_similarity_feature(content_emd_dict, trn_label_df, trn_hist_click)
    trn_w2v_sim_feat_df = get_w2v_similarity_feature(w2v_emd_dict, trn_label_df, trn_hist_click)
    trn_nearest_sim_feat_df = get_nearest_similarity_feature(i2i_sim_dict, w2v_emd_dict, content_emd_dict, trn_label_df,
                                                             trn_hist_click)
    trn_user_activate_feature_df = get_users_activate_feature(trn_label_df, trn_hist_click)
    trn_item_popular_feature_df = get_items_popular_feature(trn_label_df, trn_hist_click)

    trn_label_df = trn_label_df.merge(trn_item_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_content_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_w2v_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_nearest_sim_feat_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_user_activate_feature_df, on=["user_id", "sim_item"])
    trn_label_df = trn_label_df.merge(trn_item_popular_feature_df, on=["user_id", "sim_item"])
    trn_label_df.to_csv("Datasets/rank/trn_user_item_label_feat_df.csv")

    online_sim_dir = "output/online/similarity/"
    online_i2i_sim_dict = pickle.load(open(os.path.join(online_sim_dir, 'itemcf_i2i_sim_v2.pkl'), 'rb'))
    online_content_emd_dict = pickle.load(open(os.path.join("Datasets/online", 'item_content_vec_dict.pkl'), 'rb'))
    online_w2v_emd_dict = pickle.load(open(os.path.join("Datasets/", 'item_w2v_emb.pkl'), 'rb'))
    all_click, tst_click = get_all_click_data("online")

    tst_item_sim_feat_df = get_item_similarity_feature(online_i2i_sim_dict, tst_label_df, tst_click)
    tst_content_sim_feat_df = get_content_similarity_feature(online_content_emd_dict, tst_label_df, tst_click)
    tst_w2v_sim_feat_df = get_w2v_similarity_feature(online_w2v_emd_dict, tst_label_df, tst_click)
    tst_nearest_sim_feat_df = get_nearest_similarity_feature(online_i2i_sim_dict, online_w2v_emd_dict, online_content_emd_dict, tst_label_df,
                                                             tst_click)
    tst_user_activate_feature_df = get_users_activate_feature(tst_label_df, tst_click)
    tst_item_popular_feature_df = get_items_popular_feature(tst_label_df, all_click)

    tst_label_df = tst_label_df.merge(tst_item_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_content_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_w2v_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_nearest_sim_feat_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_user_activate_feature_df, on=["user_id", "sim_item"])
    tst_label_df = tst_label_df.merge(tst_item_popular_feature_df, on=["user_id", "sim_item"])
    tst_label_df.to_csv("Datasets/rank/tst_user_item_label_feat_df.csv")
