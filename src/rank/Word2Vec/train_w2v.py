import os

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import psutil
import pickle
import logging
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.data_process.load_data import get_all_click_data


def train_w2v_model(click_df, save_path):
    # sorted by timestamp
    click_df = click_df.sort_values('timestamp')
    # convert to string
    click_df['item_id'] = click_df['item_id'].astype(str)
    # convert to sentence
    docs = click_df.groupby(['user_id'])['item_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['item_id'].values.tolist()

    # set logging
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    # init w2v model
    model = Word2Vec(docs, size=16, sg=1, window=5, seed=2021, workers=24, min_count=1, iter=3)

    # save as dictionary
    item_w2v_emb_dict = {k: model[k] for k in click_df['item_id']}
    pickle.dump(item_w2v_emb_dict, open(save_path + 'item_w2v_emb.pkl', 'wb'))

    return item_w2v_emb_dict


if __name__ == "__main__":
    model_save_dir = "Datasets/"
    all_click, test_click = get_all_click_data("online")

    train_w2v_model(all_click, model_save_dir)