
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def get_user_item_time_dict(user_behavior):
    """
        Get the user's clicked item sequence according to the click time
        :param user_behavior: Dataframe that records the user's historical behavior
        :return: user's clicked item sequence {user1: [(item1, time1), (item2, time2)..]...}
    """
    user_behavior = user_behavior.sort_values('timestamp')
    # Normalize the timestamp to calculate the weight in association rules
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    user_behavior['timestamp'] = user_behavior[['timestamp']].apply(max_min_scaler)

    def make_item_time_pair(df):
        return list(zip(df['item_id'], df['timestamp']))

    user_item_time_df = user_behavior.groupby('user_id')['item_id', 'timestamp'].apply \
        (lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


def get_item_user_time_dict(user_behavior):
    """
    Get the user sequence of the product clicked according to time
    :param user_behavior: Dataframe that records the user's historical behavior
    :return: the user sequence {item1: [(user1, time1), (user2, time2)...]...}
    """
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['timestamp']))

    user_behavior = user_behavior.sort_values('timestamp')
    item_user_time_df = user_behavior.groupby('item_id')['user_id', 'timestamp'].apply(
        lambda x: make_user_time_pair(x)) \
        .reset_index().rename(columns={0: 'user_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['item_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict


def get_item_info_dict(item_info_df):
    """
    Get the basic attributes corresponding to the item id and save it in the form of a dictionary,
    which is convenient for the later recall stage and cold start stage to use directly
    :param item_info_df: Dataframe that records item attributes
    :return: Three dicts, corresponding to different attributes of the item
    """
    item_category_dict = dict(zip(item_info_df["item_id"], item_info_df["category"]))
    item_shop_dict = dict(zip(item_info_df["item_id"], item_info_df["shop"]))
    item_brand_dict = dict(zip(item_info_df["item_id"], item_info_df["brand"]))
    return item_category_dict, item_shop_dict, item_brand_dict


def get_hist_and_last_click(user_behavior):
    """
    Get the historical click and the last click of the current data
    :param user_behavior: Dataframe that records the user's historical behavior
    :return:
    """
    user_behavior = user_behavior.sort_values(by=['user_id', 'timestamp'])
    behavior_last_df = user_behavior.groupby('user_id').tail(1)

    # If the user has only one click, the history is empty, which will cause the user to be invisible during training.
    # At this time, it will be leaked by default.
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = user_behavior.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, behavior_last_df


def get_item_topk_click(user_behavior, k):
    """
    Get the topk items with the most clicks
    :param user_behavior: Dataframe that records the user's historical behavior
    :param k: k most popular items
    :return: Most clicked items
    """
    topk_click = user_behavior['click_article_id'].value_counts().index[:k]
    return topk_click


def get_user_activate_degree_dict(behavior):
    """
    Calculate user activity degree.
    Normalize the number of times the user clicks on the item to get the activity degree
    :param behavior:  Dataframe that records the user's historical behavior
    :return:  Dictionary, user activity degree. {user1: degree1, user2: degree2, ...}
    """
    behavior_ = behavior.groupby('user_id')['item_id'].count().reset_index()

    # User activity normalization
    mm = MinMaxScaler()
    behavior_['item_id'] = mm.fit_transform(behavior_[['item_id']])
    user_activate_degree_dict = dict(zip(behavior_['user_id'], behavior_['item_id']))

    return user_activate_degree_dict


def get_all_click_sample(data_path, sample_nums=10000):
    """
    Sampling part of the data for debugging
    :param data_path: save address of data set
    :param sample_nums: Number of samples
    :return:
    """
    all_click = pd.read_csv(data_path + 'behavior.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'item_id', 'timestamp']))
    return all_click