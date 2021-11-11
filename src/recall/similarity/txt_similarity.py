import pandas as pd
import numpy as np
import pickle
import faiss
from collections import defaultdict
from tqdm import tqdm
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import cal_cos_sim


def get_txt_sim(feature_dict, save_path, topk=20):
    """
        Similarity matrix calculation based on text embedding
        :param feature_dict: feature embedding
        :param save_path: save path
        :patam topk: the k most similar items for each item
        return Text similarity matrix

    """
    txt_emb_df = pd.DataFrame()
    txt_emb_df["feat_vec"] = feature_dict.values()
    txt_emb_df['item_id'] = list(feature_dict.keys())
    txt_emb_df["feat_vec"] = txt_emb_df["feat_vec"].apply(lambda x: x[:128])
    txt_emb_df["feat_vec"] = txt_emb_df["feat_vec"].apply(lambda x: list(x))
    # Dictionary mapping of item index and item ID
    item_idx_2_rawid_dict = dict(zip(txt_emb_df.index, txt_emb_df['item_id']))
    txt_emb_np = np.ascontiguousarray(list(txt_emb_df["feat_vec"].values), dtype=np.float32)

    # build faiss retrieval
    item_index = faiss.IndexFlatIP(txt_emb_np.shape[1])
    item_index.add(txt_emb_np)
    # The similarity query returns top k items and similarity to the vector at each index position
    sim, idx = item_index.search(txt_emb_np, topk+1)  # return list that including item itself, so topk + 1

    # Save the result of vector retrieval as the corresponding relationship of the original item id
    txt_sim_dict = defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(txt_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            if rele_idx in item_idx_2_rawid_dict.keys():
                rele_raw_id = item_idx_2_rawid_dict[rele_idx]
                txt_sim_dict[target_raw_id][rele_raw_id] = txt_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                    0) + sim_value

    # Save text similarity matrix
    pickle.dump(txt_sim_dict, open(save_path + 'txt_i2i_sim.pkl', 'wb'))

    return txt_sim_dict


if __name__ == "__main__":
    feature_dict = pickle.load(open('Datasets/online/item_content_vec_dict.pkl', 'rb'))
    save_path = "output/online/similarity/"
    get_txt_sim(feature_dict, save_path, 20)