

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