import numpy as np
import pandas as pd
import os
import random
from collections import defaultdict


def get_all_click_sample(data_path, sample_nums=10000):
    """
        Sampling part of the data from the training set for debugging
        data_path: The path of the original data
        sample_nums: Sampling number (due to the memory limitation of the machine, sampling can be done by the user here)
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click
