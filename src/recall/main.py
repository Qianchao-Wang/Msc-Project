import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.recall.recall_v1 import ItemCF
from src.recall.recall_v2 import UserCF
from src.utils.evaluation_utils import metrics_recall
from src.utils.data_utils import get_hist_and_last_click
import warnings
warnings.filterwarnings("ignore")


def main(args):
    df = pd.read_csv(os.path.join(args.data_dir, "train.txt"), sep=',')
    if args.metric_recall:
        behavior_df, trn_last_behavior_df = get_hist_and_last_click(df)
    else:
        behavior_df = df
    i2i_sim = pickle.load(open(os.path.join(args.i2i_sim_dir, 'itemcf_i2i_sim.pkl'), 'rb'))
    u2u_sim = pickle.load(open(os.path.join(args.i2i_sim_dir, 'usercf_u2u_sim.pkl'), 'rb'))
    user_recall_items_dict = defaultdict(dict)
    user_list = behavior_df['user_id'].unique()
    if args.task == "itemcf":
        itemCF = ItemCF(args=args, behavior_dataset=behavior_df, i2i_sim=i2i_sim)
        for user_id in tqdm(user_list):
            user_recall_items_dict[user_id] = itemCF.item_based_recommend(user_id)
    elif args.task == "usercf":
        userCF = UserCF(args=args, behavior_dataset=behavior_df, u2u_sim=u2u_sim)
        for user_id in tqdm(user_list):
            user_recall_items_dict[user_id] = userCF.user_based_recommend(user_id)

    if args.metric_recall:
        metrics_recall(user_recall_items_dict, trn_last_behavior_df, topk=args.recall_item_num)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="dataset/", required=True, type=str, help="The input data dir")
    parser.add_argument("--i2i_sim_dir", default="output/similarity/", type=str, help="Item similarity matrix dir")
    parser.add_argument("--u2u_sim_dir", default="output/similarity/", type=str, help="User similarity matrix dir")
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument('--sim_item_topk', type=int, default=10, help="top k items that are most similar to the current item when itemCF")
    parser.add_argument('--sim_user_topk', type=int, default=10, help="top k users that are most similar to the current user when userCF")
    parser.add_argument('--recall_item_num', type=int, default=100, help="Number of candidate sets of recalled recommended items for each strategy")

    parser.add_argument("--metric_recall", action="store_true", help="do evaluation of recall")

    # ---------------------------------------------------------------------------------
    args = parser.parse_args()

    main(args)
