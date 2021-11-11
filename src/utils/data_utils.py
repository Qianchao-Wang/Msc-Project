
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time


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
    topk_click = user_behavior['item_id'].value_counts().index[:k]
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


def cal_cos_sim(emb1, emb2):
    """
    Calculate cosine similarity between emb1 and emb2
    :param emb1:
    :param emb2:
    :return: cosine similarity
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def reduce_memory(df):
    """
    reduce the memory of Dataframe
    :param df: DataFrame,
    :return: reduce the memory of dataframe
    """
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                            (time.time()-starttime)/60))
    return df


