

def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    """
    Calculate the hit rate. hit rate = Number of users successfully hit / Total number of users
    :param user_recall_items_dict: Candidate products recalled for each user
    :param trn_last_click_df: The last item clicked by the user
    :param topk: Evaluate the first k recall candidate items
    :return:
    """
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['item_id']))
    user_num = len(user_recall_items_dict)

    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # Get the results of the first k recalled items
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)


def evaluate(predictions, answers, topk=100):
    """
    :param predictions: Recommended items returned by the model
    :param answers: The next item the user actually clicked on
    :param topk: evaluate the first topk recommended items
    :return: ndcg and hitrate
    """
    ndcg, hitrate = [], []
    for k in range(10, topk+1, 10):
        ndcg_k = 0.0
        hitrate_k = 0.0
        num_cases = 0.0
        for user_id, item_id in answers.items():
            rank = 0
            # Find the rank of the real item in the recommended list, if exist
            while rank < k and predictions[user_id][rank] != item_id:
                rank += 1
            num_cases += 1
            if rank < k:
                ndcg_k += 1.0 / np.log2(rank + 2.0)
                hitrate_k += 1

        ndcg_k /= num_cases
        hitrate_k /= num_cases
        ndcg.append(ndcg_k)
        hitrate.append(hitrate_k)

    return ndcg, hitrate
