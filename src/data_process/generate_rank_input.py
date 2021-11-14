import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.data_process.load_data import get_recall_list
from src.utils.data_utils import get_hist_and_last_click
from src.data_process.load_data import get_all_click_data
import pickle


def recall_dict_2_df(recall_list_dict):
    """

    :param recall_list_dict:
    :return:
    """

    df_row_list = [] # [user, item, score]
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append([user, item, score])

    col_names = ['user_id', 'sim_item', 'score']
    recall_items_df = pd.DataFrame(df_row_list, columns=col_names)

    return recall_items_df


def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    """

    :param recall_items_df:
    :param sample_rate:
    :return:
    """
    pos_data = recall_items_df[recall_items_df['label'] == 1]
    neg_data = recall_items_df[recall_items_df['label'] == 0]

    print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data) / len(neg_data))

    # Group sampling function
    def neg_sample_func(group_df):
        neg_num = len(group_df)
        sample_num = max(int(neg_num * sample_rate), 1)  # Ensure that there is at least one negative sample
        sample_num = min(sample_num, 2)  # Up to 5 negative samples
        return group_df.sample(n=sample_num, replace=True)

    # Negative sampling for users and ensuring that all users are in the sampled data
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    # Negative sampling for items and ensuring that all items are in the sampled data
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)

    # Combine the results of the above two samples
    neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    # Data De duplication
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'], keep='last')

    # Merge positive sample data
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)

    return data_new


def get_rank_label_df(recall_list_df, label_df, is_test=False):
    """

    :param recall_list_df:
    :param label_df:
    :param is_test:
    :return:
    """
    # The label of the test set is set to - 1
    if is_test:
        recall_list_df['label'] = -1
        return recall_list_df

    label_df = label_df.rename(columns={'item_id': 'sim_item'})
    recall_list_df_ = recall_list_df.merge(label_df[['user_id', 'sim_item', 'timestamp']], \
                                           how='left', on=['user_id', 'sim_item'])
    recall_list_df_['label'] = recall_list_df_['timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_list_df_['timestamp']

    return recall_list_df_


def get_user_recall_item_label_df(click_trn_hist, click_tst_hist, click_trn_last, recall_list_df):
    """

    :param click_trn_hist:
    :param click_tst_hist:
    :param click_trn_last:
    :param recall_list_df:
    :return:
    """
    # Get recall list of training data
    trn_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_trn_hist['user_id'].unique())]
    # Tag training data
    trn_user_item_label_df = get_rank_label_df(trn_user_items_df, click_trn_last, is_test=False)
    # Negative sampling in training data
    trn_user_item_label_df = neg_sample_recall_data(trn_user_item_label_df)

    # The test data does not need negative sampling, and all recalled items are directly labeled with - 1
    tst_user_items_df = recall_list_df[recall_list_df['user_id'].isin(click_tst_hist['user_id'].unique())]
    tst_user_item_label_df = get_rank_label_df(tst_user_items_df, None, is_test=True)
    pos_data = trn_user_item_label_df[trn_user_item_label_df["label"] == 1.0]
    neg_data = trn_user_item_label_df[trn_user_item_label_df["label"] == 0.0]
    print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data) / len(neg_data))

    return trn_user_item_label_df, tst_user_item_label_df


if __name__ == "__main__":
    data_path = 'Datasets/'
    save_path = 'Datasets/rank/offline/Candidate'
    click_trn, click_tst = get_all_click_data("offline")
    click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn)

    all_click, click_tst_hist = get_all_click_data("online")

    trn_recall_list_dict = get_recall_list('output/offline/Candidate/', multi_recall=True)
    tst_recall_list_dict = get_recall_list('output/online/Candidate/', multi_recall=True)
    trn_recall_list_df = recall_dict_2_df(trn_recall_list_dict)
    tst_recall_list_df = recall_dict_2_df(tst_recall_list_dict)
    recall_list_df = trn_recall_list_df.append(tst_recall_list_df)
    trn_user_item_label_df, tst_user_item_label_df = get_user_recall_item_label_df(
                                                                                    click_trn_hist,
                                                                                    click_tst_hist,
                                                                                    click_trn_last,
                                                                                    recall_list_df)
    trn_user_item_label_df.to_csv("Datasets/rank/trn_user_item_label_df.csv", sep=',')
    tst_user_item_label_df.to_csv("Datasets/rank/tst_user_item_label_df.csv", sep=',')
    print("Done!")