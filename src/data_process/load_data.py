import pandas as pd
import numpy as np
import time
import pickle
from src.utils.data_utils import reduce_memory
from src.utils.data_utils import get_hist_and_last_click


def get_all_click_data(mode):
    train_data_path = "Datasets/train.csv"
    train_data = pd.read_csv(
        train_data_path,
        sep=',',
        encoding='utf-8'
    )
    if mode == "offline":
        test_data = None
        return train_data, test_data
    elif mode == "online":
        test_data_path = "Datasets/test.csv"
        test_data = pd.read_csv(
            test_data_path,
            sep=',',
            encoding='utf-8'
        )
        all_click = train_data.append(test_data)
        all_click = all_click.drop_duplicates(['user_id', 'item_id', 'timestamp'], keep='last')
        return all_click, test_data


def get_answer():
    answer_path = "Datasets/answer.csv"
    answer = pd.read_csv(
        answer_path,
        sep=',',
        encoding="utf-8"
    )
    return answer


def trn_val_split(all_click_df, sample_user_nums):
    """

    :param all_click_df:
    :param sample_user_nums:
    :return:
    """
    all_click = all_click_df
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False)

    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]

    # 将验证集中的最后一次点击给抽取出来作为答案
    click_val, val_ans = get_hist_and_last_click(click_val)

    val_ans = val_ans[val_ans.user_id.isin(click_val.user_id.unique())]
    click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]

    return click_trn, click_val, val_ans


def get_trn_val_tst_data(data_path, sample_user_nums, online=False):
    """

    :param data_path:
    :param sample_user_nums:
    :param online:
    :return:
    """

    if online:
        click_trn = pd.read_csv(data_path + 'train.csv', sep=',')
        click_trn = reduce_memory(click_trn)
        click_val = None
        val_ans = None

    else:
        click_trn_data = pd.read_csv(data_path + 'train.csv', sep=',')
        click_trn_data = reduce_memory(click_trn_data)
        click_trn, click_val, val_ans = trn_val_split(click_trn_data, sample_user_nums)

    click_tst = pd.read_csv(data_path + 'test.csv')

    return click_trn, click_val, click_tst, val_ans


def get_recall_list(save_dir, single_recall_model=None, multi_recall=False):
    """

    :param save_dir:
    :param single_recall_model:
    :param multi_recall:
    :return:
    """
    if multi_recall:
        return pickle.load(open(save_dir + 'final_recall_items_dict.pkl', 'rb'))

    if single_recall_model == 'itemcf':
        return pickle.load(open(save_dir + 'itemcf_recall_candidate.pkl', 'rb'))
    elif single_recall_model == 'usercf':
        return pickle.load(open(save_dir + 'usercf_recall_candidate.pkl', 'rb'))
    elif single_recall_model == 'srgnn':
        return pickle.load(open(save_dir + 'srgnn_recall_candidate.pkl', 'rb'))