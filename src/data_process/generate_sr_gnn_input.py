from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import pickle
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_user_item_time_dict
from src.data_process.load_data import get_all_click_data
from src.data_process.load_feat import obtain_entire_item_feat_df


def weighted_agg_content(hist_item_id_list, item_content_vec_dict):
    # weighted user behavior sequences to obtain user initial embedding
    weighted_vec = np.zeros(128*2)
    hist_num = len(hist_item_id_list)
    sum_weight = 0.0
    for loc, (i, t) in enumerate(hist_item_id_list):
        loc_weight = 0.9 ** (hist_num - loc)
        if i in item_content_vec_dict:
            sum_weight += loc_weight
            weighted_vec += loc_weight*item_content_vec_dict[i]
    if sum_weight != 0:
        weighted_vec /= sum_weight
        txt_item_feat_np = weighted_vec[0:128] / np.linalg.norm(weighted_vec[0:128])
        img_item_feat_np = weighted_vec[128:] / np.linalg.norm(weighted_vec[128:])
        weighted_vec = np.concatenate([txt_item_feat_np,  img_item_feat_np])
    else:
        print('zero weight...')
    return weighted_vec


def init_item_user_embedding(item_content_vec_dict, mode):
    all_click, test_click = get_all_click_data(mode)
    user_item_time_hist_dict = get_user_item_time_dict(all_click)
    sr_gnn_dir = "src/models/sr_gnn/input"
    lbe = LabelEncoder()
    lbe.fit(all_click['item_id'].astype(str))
    item_raw_id2_idx_dict = dict(zip(lbe.classes_, lbe.transform(lbe.classes_) + 1, ))  # get dictionary
    item_cnt = len(item_raw_id2_idx_dict)
    print(item_cnt)

    lbe = LabelEncoder()
    lbe.fit(all_click['user_id'].astype(str))
    user_raw_id2_idx_dict = dict(zip(lbe.classes_, lbe.transform(lbe.classes_) + 1, ))  # get dictionary
    user_cnt = len(user_raw_id2_idx_dict)
    print(user_cnt)

    item_embed_np = np.zeros((item_cnt + 1, 256))
    for raw_id, idx in item_raw_id2_idx_dict.items():
        vec = item_content_vec_dict[int(raw_id)]
        item_embed_np[idx, :] = vec
    np.save(open(sr_gnn_dir + '/item_embed_mat.npy', 'wb'), item_embed_np)

    user_embed_np = np.zeros((user_cnt + 1, 256))
    for raw_id, idx in user_raw_id2_idx_dict.items():
        hist = user_item_time_hist_dict[int(raw_id)]
        vec = weighted_agg_content(hist, item_content_vec_dict)
        user_embed_np[idx, :] = vec
    np.save(open(sr_gnn_dir + '/user_embed_mat.npy', 'wb'), user_embed_np)

    return item_raw_id2_idx_dict, user_raw_id2_idx_dict


def get_item_sequence():
    all_click, test_click = get_all_click_data("online")
    full_user_item_dict = get_user_item_time_dict(all_click)
    print(len(full_user_item_dict))
    # 4.1 train sequences
    train_user_hist_seq_dict = {}
    for u, hist_seq in full_user_item_dict.items():
        if len(hist_seq) > 1:
            train_user_hist_seq_dict[u] = hist_seq
    train_users = train_user_hist_seq_dict.keys()
    print(len(train_user_hist_seq_dict))
    # 4.2 validate sequences and infer sequences
    test_users = test_click['user_id'].unique()
    test_user_hist_seq_dict = {}
    infer_user_hist_seq_dict = {}
    for test_u in test_users:
        if test_u not in full_user_item_dict:
            print('test-user={} not in train/test data'.format(test_u))
            continue
        if len(full_user_item_dict[test_u]) > 1:
            test_user_hist_seq_dict[test_u] = full_user_item_dict[test_u]
            if test_u in train_user_hist_seq_dict:
                if len(train_user_hist_seq_dict[test_u][:-1]) > 1:
                    train_user_hist_seq_dict[test_u] = train_user_hist_seq_dict[test_u][: -1]  # last one not train, use just for test
                else:
                    del train_user_hist_seq_dict[test_u]

        infer_user_hist_seq_dict[test_u] = full_user_item_dict[test_u]

    print(len(train_user_hist_seq_dict))
    print(len(test_user_hist_seq_dict))
    print(len(infer_user_hist_seq_dict))
    return train_user_hist_seq_dict, test_user_hist_seq_dict, infer_user_hist_seq_dict, train_users, test_users


def gen_data(sr_gnn_dir, is_attach_user=False):
    processed_item_feat_df = pd.read_csv("Datasets/processed_item_feat.csv", sep=",")
    item_content_vec_dict = pickle.load(open('Datasets/item_content_vec_dict.pkl'), 'rb')
    item_raw_id2_idx_dict, user_raw_id2_idx_dict = init_item_user_embedding(item_content_vec_dict, "online")
    train_user_hist_seq_dict, test_user_hist_seq_dict, infer_user_hist_seq_dict, train_users, test_users = get_item_sequence()
    with open(sr_gnn_dir + '/train_item_seq.txt', 'w') as f_seq, \
            open(sr_gnn_dir + '/train_user_sess.txt', 'w') as f_user:
        for u in train_users:
            u_idx = user_raw_id2_idx_dict[str(u)]
            hist_item_time_seq = train_user_hist_seq_dict[u]
            hist_item_seq = [str(item_raw_id2_idx_dict[str(item)]) for item, time in hist_item_time_seq]
            if is_attach_user:
                hist_item_seq_sess = [str(u_idx), ] + hist_item_seq
            else:
                hist_item_seq_sess = hist_item_seq

            hist_item_seq_str = " ".join(hist_item_seq_sess)
            f_seq.write(hist_item_seq_str + '\n')

            # infer
            if is_attach_user:
                hist_item_user_sess = [str(u), str(u_idx)] + hist_item_seq
            else:
                hist_item_user_sess = [str(u), ] + hist_item_seq
            hist_item_user_sess_str = " ".join(hist_item_user_sess)
            f_user.write(hist_item_user_sess_str + '\n')

    with open(sr_gnn_dir + '/test_item_seq.txt', 'w') as f_seq, open(sr_gnn_dir + '/test_user_sess.txt',
                                                                          'w') as f_user:
        for u in test_users:
            # test
            if u in test_user_hist_seq_dict:
                u_idx = user_raw_id2_idx_dict[str(u)]
                hist_item_time_seq = test_user_hist_seq_dict[u]
                hist_item_seq = [str(item_raw_id2_idx_dict[str(item)]) for item, time in hist_item_time_seq]

                if is_attach_user:
                    hist_item_seq_sess = [str(u_idx), ] + hist_item_seq
                else:
                    hist_item_seq_sess = hist_item_seq

                hist_item_seq_str = " ".join(hist_item_seq_sess)
                f_seq.write(hist_item_seq_str + '\n')

            if u in infer_user_hist_seq_dict:
                hist_item_time_seq = infer_user_hist_seq_dict[u]
                hist_item_seq = [str(item_raw_id2_idx_dict[str(item)]) for item, time in hist_item_time_seq]

                if is_attach_user:
                    hist_item_user_sess = [str(u), str(u_idx)] + hist_item_seq
                else:
                    hist_item_user_sess = [str(u), ] + hist_item_seq

                hist_item_user_sess_str = " ".join(hist_item_user_sess)
                f_user.write(hist_item_user_sess_str + '\n')

    with open(sr_gnn_dir + '/item_lookup.txt', 'w') as f_item_map:
        for raw_id, idx in item_raw_id2_idx_dict.items():
            f_item_map.write("{} {}\n".format(idx, raw_id))


def data_augmentation(sr_gnn_dir, is_attach_user=False):
    np.random.seed(2021)
    count = 0
    max_len = 10
    tmp_max = 0
    with open(sr_gnn_dir + '/train_item_seq.txt', 'r') as f_in, open(
            sr_gnn_dir + '/train_item_seq_enhanced.txt', 'w') as f_out:
        for line in f_in:
            row = line.strip().split()

            if is_attach_user:
                uid = row[0]
                iids = row[1:]
            else:
                iids = row

            end_step_1 = max(2, np.random.poisson(4))
            end_step_2 = len(iids) + 1

            if end_step_2 > end_step_1:
                for i in range(end_step_1, end_step_2):
                    count += 1
                    start_end = max(i - max_len, 0)
                    tmp_max = max(tmp_max, len(iids[start_end: i]))
                    sampled_seq = iids[start_end: i]

                    if is_attach_user:
                        sampled_seq = [str(uid), ] + sampled_seq

                    f_out.write(' '.join(sampled_seq) + '\n')
            else:
                count += 1
                f_out.write(line)
    print("Done, Output Lines: {}".format(count))
    print(tmp_max)


if __name__ == "__main__":
    sr_gnn_dir = "src/models/sr_gnn/input"
    gen_data(sr_gnn_dir, is_attach_user=True)
    data_augmentation(sr_gnn_dir, is_attach_user=True)
