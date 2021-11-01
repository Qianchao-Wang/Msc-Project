import os

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import psutil



def train_w2v_model(model_save_dir, sentences):
    print('Start...')
    model = Word2Vec(
        sentences,
        size=16,
        alpha=0.1,
        window=999999,
        min_count=1,
        workers=psutil.cpu_count(),
        compute_loss=True,
        iter=5,
        hs=0,
        sg=1,
        seed=42
    )
    print(model.max_final_vocab)
    model.wv.save_word2vec_format(model_save_dir, binary=False)
    print("Finished!")


if __name__ == "__main__":
    model_save_dir = "output/word2vec/"
    data_train = pd.read_csv(
        "./behavior_underpose_train.csv"
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
    )
    data_test = pd.read_csv(
        "./behavior_underpose_test.csv"
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
    )
    data_train = data_train.append(data_test)
    data_train = data_train.drop_duplicates(["user_id", "item_id", "timestamp"], keep="last")
    data_train = data_train.sort_values('timestamp')
    data_ = data_train.groupby(['user_id'])['item_id'].agg(lambda x: ','.join(list(x))).reset_index()
    list_data = list(data_['item_id'].map(lambda x: x.split(',')))
    train_w2v_model(model_save_dir, list_data)