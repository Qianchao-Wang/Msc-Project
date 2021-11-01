import numpy as np
from collections import defaultdict
import logging
import json
import os
import warnings
from src.models.sr_gnn.train.data_loader import DataLoader
from src.models.sr_gnn.modeling.SR_GNN import SRGNN
from tqdm import tqdm, trange
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args):

        self.args = args
        self._at = [50]
        self.node_count = args.node_count
        self.checkpoint_path = args.checkpoint_path

    def train(self):
        # get some configs
        train_path = self.args.train_input
        max_test_batch = self.args.max_test_batch
        max_len = self.args.max_len
        has_uid = self.args.has_uid  # whether has uid in the beginning of
        sq_max_len = self.args.sq_max_len
        train = DataLoader(self.args.train_input, max_len=max_len, has_uid=has_uid, sq_max_len=sq_max_len)
        test_path = self.args.test_input
        test = DataLoader(test_path, True if max_test_batch else False, max_len=max_len,
                          has_uid=has_uid, sq_max_len=sq_max_len) if test_path else None

        epochs = self.args.epochs
        batch_logging_step = self.args.batch_logging_step
        save_step = self.args.save_step
        batch_size = self.args.batch_size
        early_stop_epochs = self.args.early_stop_epochs
        if 'lr_dc' in self.args:
            self.args.lr_dc = self.args.lr_dc * np.ceil(train.count / batch_size)

        if 'node_weight' in self.args and self.args.node_weight.lower() in ('yes', 'true', 't', 'y', '1'):
            nw = train.get_node_weight(self.node_count)
            np.save(os.path.join(os.path.dirname(train_path), 'node_weight'), nw)
        kwargs = {k: v for k, v in self.args.__dict__.items()}
        logger.info('Train: {}'.format(kwargs))
        # init the model
        model = SRGNN(args=self.args, node_count=self.node_count + 1, restore_path=self.checkpoint_path)

        global_step = int(model.run_step())
        best_result, best_epoch = [[0] * len(self._at), [0] * len(self._at)], [[0] * len(self._at), [0] * len(self._at)]  # only @20
        early_stop_counter = 0
        print_nums = lambda x: ' '.join(map(lambda num: '{:.4f}'.format(num), x))
        # start training
        train_iterator = trange(int(epochs), desc="Epoch")
        for epoch in range(train_iterator):
            slices = train.generate_batch(batch_size)
            ll = []
            logger.info('Total Batch: {}'.format(len(slices)))
            for index, i in enumerate(slices):
                adj_in, adj_out, graph_item, last_node_id, attr_dict = train.get_slice(i)
                input_session = (adj_in, adj_out, graph_item, last_node_id, attr_dict['next_item'])
                loss = model.run_train(input_session, attr_dict['node_pos'] if sq_max_len is not None else None)
                ll.append(loss)
                global_step += 1
                if index % batch_logging_step == 0:
                    logger.info('Batch {}, Loss: {:.5f}'.format(index, np.mean(ll)))
                if save_step and self.checkpoint_path and global_step % save_step == 0:
                    model.save(self.checkpoint_path, global_step)
            if self.checkpoint_path and not test:
                model.save(path=self.checkpoint_path, global_step=global_step)
            if not test:
                logger.info('Epoch: {} Train Loss: {:.4f}'.format(epoch, np.mean(ll)))
                continue

            slices = test.generate_batch(batch_size)
            test_loss_ = []
            hit = [[] for _ in self._at]
            mrrs = [[] for _ in self._at]
            for ii, i in enumerate(slices):
                adj_in, adj_out, graph_items, last_node_id, attr_dict = test.get_slice(i)
                input_session = (adj_in, adj_out, graph_items, last_node_id, attr_dict['next_item'])
                loss, scores = model.run_eval(input_session, attr_dict['node_pos'] if sq_max_len is not None else None)
                score_top100 = np.argsort(scores, 1)[:, -100:]
                test_loss_.append(loss)
                recall_a = [self.calc_recall(score_top100, attr_dict['next_item'], k) for k in self._at]
                mrr_a = [self.calc_mrr(score_top100, attr_dict['next_item'], k) for k in self._at]
                recall = map(lambda x: x[0], recall_a)
                mrr = map(lambda x: x[0], mrr_a)
                logger.info('Test Loss: {:.4f}  @{}, Recall: {}  MRR: {}'.format(loss, ' '.join(map(str, self._at)),
                                                                                 print_nums(recall), print_nums(mrr)))
                for pos, _ in enumerate(self._at):
                    hit[pos] += recall_a[pos][1]
                    mrrs[pos] += mrr_a[pos][1]
                if max_test_batch and ii >= max_test_batch - 1: break
            epoch_hits = np.mean(hit, axis=1)
            epoch_mrr = np.mean(mrrs, axis=1)
            is_improve = 0
            for i in range(len(self._at)):
                if epoch_hits[i] > best_result[0][i]:
                    best_result[0][i] = epoch_hits[i]
                    best_epoch[0][i] = epoch
                    is_improve = 1
                if epoch_mrr[i] > best_result[1][i]:
                    best_result[1][i] = epoch_mrr[i]
                    best_epoch[1][i] = epoch
                    is_improve = 1
            logger.info('Epoch: {} Train Loss: {:.4f} Test Loss: {:.4f} Recall: {} MRR: {}'.format(epoch, np.mean(ll),
                                                                                                   np.mean(test_loss_),
                                                                                                   print_nums(
                                                                                                       epoch_hits),
                                                                                                   print_nums(
                                                                                                       epoch_mrr)))
            logger.info('Best Recall and MRR: {},  {}  Epoch: {},  {}'.format(print_nums(best_result[0]),
                                                                              print_nums(best_result[1]),
                                                                              ' '.join(map(str, best_epoch[0])),
                                                                              ' '.join(map(str, best_epoch[1]))))
            if self.checkpoint_path and is_improve == 1:
                model.save(self.checkpoint_path, global_step)
            early_stop_counter += 1 - is_improve
            if early_stop_epochs and early_stop_counter >= early_stop_epochs:
                logger.info('After {} epochs not improve, early stop'.format(early_stop_counter))
                break
        logger.info('Best Recall and MRR: {},  {}  Epoch: {},  {}'.format(print_nums(best_result[0]),
                                                                          print_nums(best_result[1]),
                                                                          ' '.join(map(str, best_epoch[0])),
                                                                          ' '.join(map(str, best_epoch[1]))))

    def run_eval(self, eval_path, node_count, checkpoint_path):
        max_test_batch = self.args.max_test_batch
        max_len = self.args.max_len
        has_uid = self.args.has_uid
        sq_max_len = self.args.sq_max_len
        eval_data = DataLoader(eval_path, True if max_test_batch else False, max_len=max_len,
                               has_uid=has_uid, sq_max_len=sq_max_len)

        batch_size = self.args.batch_size
        model = SRGNN(node_count + 1, checkpoint_path, self.args)

        slices = eval_data.generate_batch(batch_size)
        logger.info('Total Batch: {}'.format(len(slices)))
        test_loss_ = []

        hit = [[] for _ in self._at]
        mrrs = [[] for _ in self._at]
        print_nums = lambda x: ' '.join(map(lambda num: '{:.4f}'.format(num), x))
        for index, i in enumerate(slices):
            adj_in, adj_out, graph_item, last_node_id, attr_dict = eval_data.get_slice(i)
            input_session = (adj_in, adj_out, graph_item, last_node_id, attr_dict['next_item'])
            loss, scores = model.run_eval(input_session, attr_dict['node_pos'] if sq_max_len is not None else None)
            score_top100 = np.argsort(scores, 1)[:, -100:]
            test_loss_.append(loss)
            recall_a = [self.calc_recall(score_top100, attr_dict['next_item'], k) for k in self._at]
            mrr_a = [self.calc_mrr(score_top100, attr_dict['next_item'], k) for k in self._at]
            recall = map(lambda x: x[0], recall_a)
            mrr = map(lambda x: x[0], mrr_a)
            logger.info(
                'Eval Loss: {:.4f}  @{}, Recall: {}  MRR: {}'.format(loss, ' '.join(map(str, self._at)), print_nums(recall),
                                                                     print_nums(mrr)))
            for pos, _ in enumerate(self._at):
                hit[pos] += recall_a[pos][1]
                mrrs[pos] += mrr_a[pos][1]
            if max_test_batch and index >= max_test_batch - 1: break
        logger.info(
            'Total: Eval Loss: {:.4f} Recall: {} MRR: {}'.format(np.mean(test_loss_), print_nums(np.mean(hit, axis=1)),
                                                                 print_nums(np.mean(mrrs, axis=1))))

    def run_recommend(self, session_input, checkpoint_path, node_count, output_path):
        max_len = self.args.max_len
        has_uid = self.args.has_uid
        sq_max_len = self.args.sq_max_len
        session_data = DataLoader(session_input, False, False, True,
                                  max_len=max_len, has_uid=has_uid, sq_max_len=sq_max_len)
        batch_size = self.args.batch_size
        if 'node_weight' in self.args and self.args.node_weight.lower() in ('yes', 'true', 't', 'y', '1') \
                and self.args.node_weight_trainable:
            nw = np.zeros([node_count + 1], np.float32)
            self.args.node_weight = nw
        model = SRGNN(args=self.args, node_count=self.node_count + 1, restore_path=checkpoint_path)
        id_lookup_file = self.args.item_lookup
        if id_lookup_file:
            item_id = {}
            with open(id_lookup_file, 'r') as f:
                for line in f:
                    num_id, vid = line.split()
                    item_id[int(num_id)] = vid
        slices = session_data.generate_batch(batch_size)
        logger.info('Total Batch: {}'.format(len(slices)))
        total_users = 0
        remove_duplicates = self.args.remove_duplicates
        if remove_duplicates is None: remove_duplicates = True
        rec_count = self.args.rec_count
        rec_extra_count = self.args.rec_extra_count
        topks = rec_count + rec_extra_count + 1
        with open(output_path, 'w') as f:
            for i, batch in enumerate(slices):
                adj_in, adj_out, graph_item, last_node_id, attr_dict = session_data.get_slice(batch)
                input_sessions = (adj_in, adj_out, graph_item, last_node_id)
                scores = model.run_predict(input_sessions, attr_dict['node_pos'] if sq_max_len is not None else None)
                score_topks = np.argsort(scores, 1)[:, :-topks:-1]
                topk_logits = np.asarray([scores[k][score_topks[k]] for k in range(len(score_topks))])
                score_topks = score_topks + 1
                headers = attr_dict['header']
                for j in range(len(score_topks)):
                    items = score_topks[j].tolist()
                    logits = np.exp(topk_logits[j] - np.max(topk_logits[j]))
                    ls = np.sum(logits)
                    logits = (logits / ls).tolist()
                    item_score = list(zip(items, logits))
                    if remove_duplicates:
                        session_items = set(graph_item[j])
                        item_score = list(filter(lambda x: x[0] not in session_items, item_score))
                    if id_lookup_file:
                        item_score = list(map(lambda x: (item_id[x[0]], x[1]), item_score))
                    item_score = item_score[:rec_count]
                    f.write(json.dumps([headers[j], item_score]) + '\n')
                    total_users += 1

                logger.info('Batch {} Finished, users: {}'.format(i, total_users))

            logger.info('Recommend Finished, users: {}'.format(total_users))

    def run_node_embedding(self, checkpoint_path, node_count, output_path):
        if 'node_weight' in self.args and self.args.node_weight.lower() in ('yes', 'true', 't', 'y', '1'):
            nw = np.zeros([node_count + 1], np.float32)
            self.args.node_weight = nw
        model = SRGNN(node_count + 1, checkpoint_path, self.args)
        np.save(output_path, model.run_embedding()[0])

    def calc_recall(self, scores, targets, topk=10):
        is_hit = []
        for s, t in zip(scores, targets):
            is_hit.append(np.isin(t - 1, s[-topk:]))
        return np.mean(is_hit), is_hit

    def calc_mrr(self, scores, targets, topk=10):
        mrr = []
        for s, t in zip(scores, targets):
            pos = np.where(s[-topk:] == t - 1)[0]
            if len(pos) > 0:
                mrr.append(1 / (topk - pos[0]))
            else:
                mrr.append(0)
        return np.mean(mrr), mrr