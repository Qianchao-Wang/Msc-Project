
import sys
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.utils.data_utils import get_user_item_time_dict
from src.utils.data_utils import get_item_topk_click


class UserCF(object):
    def __init__(self, args, behavior_dataset=None, u2u_sim=None, item_topk_click=None):
        self.args = args
        self.behavior = behavior_dataset
        self.user_item_time_dict = get_user_item_time_dict(self.behavior)
        # Get the list of items with the most clicks used for the user's candidate items completion
        self.item_topk_click = item_topk_click
        self.u2u_sim = u2u_sim

    def user_based_recommend(self, user_id):
        """
            Recall based on user collaborative filtering
            :param user_id: user id
            return: Recommended candidate set [(item1, score1), (item2, score2)...]
        """
        # Historical interaction
        user_item_time_list = self.user_item_time_dict[user_id]  # [(item1, time1), (item2, time2)..]
        # There are multiple interactions between a user and a certain item, here we have to remove the duplicates
        user_hist_items = set([i for i, t in user_item_time_list])

        items_rank = {}
        for sim_u, sim_uv in sorted(self.u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:self.args.sim_user_topk]:
            if sim_u in self.user_item_time_dict.keys():
                for loc, (i, click_time) in enumerate(self.user_item_time_dict[sim_u]):
                    if i in user_hist_items:
                        continue
                    items_rank.setdefault(i, 0)

                    loc_weight = (0.9 ** (len(user_hist_items) - loc))

                    items_rank[i] += loc_weight * sim_uv

        # if candidate items are less than recall_item_num, complete  with popular items
        if len(items_rank) < self.args.recall_item_num:
            for i, item in enumerate(self.item_topk_click):
                if item in items_rank.items():  # The filled item should not be in the original list
                    continue
                items_rank[item] = - i - 1
                if len(items_rank) == self.args.recall_item_num:
                    break

        items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:self.args.recall_item_num]
        return items_rank
