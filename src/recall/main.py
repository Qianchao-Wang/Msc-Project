import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.recall.recall_itemcf import ItemCF
from src.recall.recall_usercf import UserCF
from src.utils.evaluation_utils import metrics_recall
from src.utils.data_utils import get_hist_and_last_click
from src.data_process.load_data import get_all_click_data
from src.utils.data_utils import get_item_topk_click
from src.recall.combine_recall_results import combine_recall_results
import warnings
warnings.filterwarnings("ignore")


def get_srgnn_recall_candidate(candidate_file):
    rec_user_item_dict = {}
    with open(candidate_file) as f:
        for line in f:
            try:
                row = eval(line)
                uid = row[0]
                iids = row[1]
                iids = [(int(iid), float(score)) for iid, score in iids]
                iids = sorted(iids, key=lambda x: x[1], reverse=True)
                rec_user_item_dict[int(uid)] = iids
            except:
                print(line)
    print('read sr-gnn done, num={}'.format(len(rec_user_item_dict)))
    return rec_user_item_dict


def main(args):
    all_click, test_click = get_all_click_data(args.mode)

    if args.mode == "online":
        item_topk_click = get_item_topk_click(all_click, k=200)
        answer = pd.read_csv("Datasets/answer.csv", sep=',', encoding='utf-8')
        user_list = test_click['user_id'].unique()

    elif args.mode == "offline":
        user_hist_click, answer = get_hist_and_last_click(all_click)
        item_topk_click = get_item_topk_click(user_hist_click, k=200)
        user_list = user_hist_click['user_id'].unique()
        all_click = user_hist_click

    else:
        exit(-1)
    if args.task == "itemcf":
        user_recall_items_dict = defaultdict(dict)
        i2i_sim = pickle.load(open(os.path.join(args.i2i_sim_dir, 'itemcf_i2i_sim.pkl'), 'rb'))
        itemCF = ItemCF(args=args, behavior_dataset=all_click, i2i_sim=i2i_sim, item_topk_click=item_topk_click)
        for user_id in tqdm(user_list):
            user_recall_items_dict[user_id] = itemCF.item_based_recommend(user_id)
        pickle.dump(user_recall_items_dict, open(args.save_dir + '{}_recall_candidate.pkl'.format(args.task), 'wb'))
    elif args.task == "itemcf_baseline":
        user_recall_items_dict = defaultdict(dict)
        i2i_sim = pickle.load(open(os.path.join(args.i2i_sim_dir, 'itemcf_i2i_sim_v1.pkl'), 'rb'))
        itemCF = ItemCF(args=args, behavior_dataset=all_click, i2i_sim=i2i_sim, item_topk_click=item_topk_click)
        for user_id in tqdm(user_list):
            user_recall_items_dict[user_id] = itemCF.item_based_recommend(user_id)
        pickle.dump(user_recall_items_dict, open(args.save_dir + '{}_recall_candidate_v1.pkl'.format(args.task), 'wb'))
        
    elif args.task == "usercf":
        user_recall_items_dict = defaultdict(dict)
        u2u_sim = pickle.load(open(os.path.join(args.u2u_sim_dir, 'usercf_u2u_sim.pkl'), 'rb'))
        userCF = UserCF(args=args, behavior_dataset=all_click, u2u_sim=u2u_sim, item_topk_click=item_topk_click)
        for user_id in tqdm(user_list):
            user_recall_items_dict[user_id] = userCF.user_based_recommend(user_id)
        pickle.dump(user_recall_items_dict, open(args.save_dir + 'usercf_recall_candidate.pkl', 'wb'))
    elif args.task == "usercf_baseline":
        user_recall_items_dict = defaultdict(dict)
        u2u_sim = pickle.load(open(os.path.join(args.u2u_sim_dir, 'usercf_u2u_sim_v1.pkl'), 'rb'))
        userCF = UserCF(args=args, behavior_dataset=all_click, u2u_sim=u2u_sim, item_topk_click=item_topk_click)
        for user_id in tqdm(user_list):
            user_recall_items_dict[user_id] = userCF.user_based_recommend(user_id)
        pickle.dump(user_recall_items_dict, open(args.save_dir + 'usercf_recall_candidate_v1.pkl', 'wb'))

    elif args.task == "srgnn":
        user_recall_items_dict = get_srgnn_recall_candidate(args.candidate_file)
        pickle.dump(user_recall_items_dict, open(args.save_dir + 'srgnn_recall_candidate.pkl', 'wb'))
    metrics_recall(user_recall_items_dict, answer, topk=args.recall_item_num)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    task = ['itemcf', 'usercf', 'srgnn', 'itemcf_baseline', 'usercf_baseline']
    parser.add_argument("--task", default=None, required=True, type=str, choices=task, help="The name of the task to train")
    parser.add_argument("--mode", default=None, required=True, type=str, help="online or offline")
    parser.add_argument("--data_dir", default="Datasets/", required=True, type=str, help="The input data dir")
    parser.add_argument("--i2i_sim_dir", default="output/online/similarity/", type=str, help="Item similarity matrix dir")
    parser.add_argument("--u2u_sim_dir", default="output//online/similarity/", type=str, help="User similarity matrix dir")
    parser.add_argument("--save_dir", default="output/Candidate/", type=str, help="result dir")
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument('--sim_item_topk', type=int, default=10, help="top k items that are most similar to the current item when itemCF")
    parser.add_argument('--sim_user_topk', type=int, default=10, help="top k users that are most similar to the current user when userCF")
    parser.add_argument('--recall_item_num', type=int, default=100, help="Number of candidate sets of recalled recommended items for each strategy")
    parser.add_argument("--candidate_file", default="output/online/Candidate/srgnn_recall_v1.txt", type=str, help="file path of recommend results generated by SR-GNN")

    parser.add_argument("--metric_recall", action="store_true", help="do evaluation of recall")

    # ---------------------------------------------------------------------------------
    args = parser.parse_args()

    main(args)
