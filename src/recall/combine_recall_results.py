from tqdm import tqdm
import pickle
import os


def norm_user_recall_items_sim(sorted_item_list):

    if len(sorted_item_list) < 2:
        return sorted_item_list

    min_sim = sorted_item_list[-1][1]
    max_sim = sorted_item_list[0][1]

    norm_sorted_item_list = []
    for item, score in sorted_item_list:
        if max_sim > 0:
            norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
        else:
            norm_score = 0.0
        norm_sorted_item_list.append((item, norm_score))

    return norm_sorted_item_list


def combine_recall_results(user_multi_recall_dict, weight_dict=None, save_path="output/Candidate/"):
    final_recall_items_dict = {}

    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # When calculating the final recall result, set a weight for each recall result
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        for user_id, sorted_item_list in user_recall_items.items():
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {}

    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)

    # save the result
    pickle.dump(final_recall_items_dict_rank, open(os.path.join(save_path, 'final_recall_items_dict.pkl'), 'wb'))

    return final_recall_items_dict_rank