import pandas as pd
import numpy as np
import pickle
from src.utils.data_utils import reduce_memory
from src.utils.data_utils import get_hist_and_last_click


def get_all_click_data(mode):
    """

    :param mode:
    :return:
    """
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

