import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
import argparse
warnings.filterwarnings('ignore')


def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


def lgb_classifier_main(trn_final_df, tst_final_df, save_path):
    k_fold = 5
    trn_df = trn_final_df
    user_set = get_kfold_users(trn_df, n=k_fold)

    score_list = []
    score_df = trn_df[['user_id', 'sim_item', 'label']]
    sub_preds = np.zeros(tst_final_df.shape[0])
    lgb_cols = ["score", "max_i2i_sim", "mean_i2i_sim", "var_i2i_sim", "max_content_sim", "mean_content_sim",
                "var_content_sim", "max_w2v_sim", "mean_w2v_sim", "var_w2v_sim", "item_sim_1", "item_sim_2",
                "item_sim_3", "content_sim_1", "content_sim_2", "content_sim_3", "w2v_sim_1", "w2v_sim_2",
                "w2v_sim_3", "user_activate"]

    # 五折交叉验证，并将中间结果保存用于staking
    for n_fold, valid_user in enumerate(user_set):
        train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
        valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

        # 模型及参数的定义
        lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                               max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7,
                                               subsample_freq=1,
                                               learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=16,
                                               verbose=10)
        # 训练模型
        lgb_Classfication.fit(train_idx[lgb_cols], train_idx['label'], eval_set=[(valid_idx[lgb_cols], valid_idx['label'])],
                              eval_metric=['auc', ], early_stopping_rounds=50, )

        # 预测验证集结果
        valid_idx['pred_score'] = lgb_Classfication.predict_proba(valid_idx[lgb_cols],
                                                                  num_iteration=lgb_Classfication.best_iteration_)[:, 1]

        # 对输出结果进行归一化 分类模型输出的值本身就是一个概率值不需要进行归一化
        # valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

        valid_idx.sort_values(by=['user_id', 'pred_score'])
        valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

        # 将验证集的预测结果放到一个列表中，后面进行拼接
        score_list.append(valid_idx[['user_id', 'sim_item', 'pred_score', 'pred_rank']])

        # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均

        sub_preds += lgb_Classfication.predict_proba(tst_final_df[lgb_cols],
                                                     num_iteration=lgb_Classfication.best_iteration_)[:, 1]

    score_df_ = pd.concat(score_list, axis=0)
    score_df = score_df.merge(score_df_, how='left', on=['user_id', 'sim_item'])
    # 保存训练集交叉验证产生的新特征
    score_df[['user_id', 'sim_item', 'pred_score', 'pred_rank', 'label']].to_csv(
        save_path + 'trn_lgb_cls_feats.csv', index=False)

    # 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
    tst_final_df['pred_score'] = sub_preds / k_fold
    tst_final_df['pred_score'] = tst_final_df['pred_score'].transform(lambda x: norm_sim(x))
    tst_final_df.sort_values(by=['user_id', 'pred_score'])
    tst_final_df['pred_rank'] = tst_final_df.groupby(['user_id'])[
        'pred_score'].rank(ascending=False, method='first')

    # 保存测试集交叉验证的新特征
    tst_final_df[['user_id', 'sim_item', 'pred_score', 'pred_rank']].to_csv(
        save_path + 'tst_lgb_cls_feats.csv', index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    task = ['itemcf', 'usercf', 'srgnn']
    parser.add_argument("--task", default=None, required=True, type=str, choices=task, help="The name of the task")
    args = parser.parse_args()
    
    trn_df_path = "Datasets/rank/{}/trn_user_item_label_feat_df.csv".format(args.task)
    tst_df_path = "Datasets/rank/{}/tst_user_item_label_feat_df.csv".format(args.task)
    trn_final_df = pd.read_csv(trn_df_path)
    tst_final_df = pd.read_csv(tst_df_path)
    
    save_path = "output/rank/{}/".format(args.task)
    lgb_classifier_main(trn_final_df, tst_final_df, save_path)