import logging
import argparse
import sys, os
sys.path.append("/content/drive/My Drive/Msc Project")  # if run in colab
from src.models.sr_gnn.train.trainer import Trainer
import warnings
warnings.filterwarnings("ignore")


def find_checkpoint_path(checkpoint_prefix='srgnn_model.ckpt'):
    checkpoint_dir = "src/models/sr_gnn/output/checkpoint"
    step_max = 0
    re_cp = re.compile("{}-(\d+)\.".format(checkpoint_prefix))
    for file in os.listdir(checkpoint_dir):
        so = re_cp.search(file)
        if so:
            step = int(so.group(1))
            step_max = step if step > step_max else step_max
    checkpoint_path = '{}/{}-{}'.format(checkpoint_dir, checkpoint_prefix, step_max)
    print('CheckPoint: {}'.format(checkpoint_path))
    return checkpoint_path


def main(args):
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", level=logging.INFO)
    logger = logging.getLogger("main")
    trainer = Trainer(args)

    if args.task == 'train':
        if not args.train_input:
            logger.error("Arg Error: --train_input")
            exit(-1)
        trainer.train(args.train_input)
    elif args.task == 'eval':
        if not args.eval_input:
            logger.error("Arg Error: --eval_input")
            exit(-1)
        trainer.run_eval(args.eval_input, args.node_count, args.heckpoint)
    elif args.task == 'recommend':
        if not args.recommend_output or not args.session_input:
            logger.error("Arg Error: --recommend_output/--session_input")
            exit(-1)
        trainer.run_recommend(session_input=args.session_input, checkpoint_path=args.checkpoint_path,
                              node_count=args.node_count, output_path=args.recommend_output)
    elif args.task == 'node_embedding':
        if not args.embedding_output:
            logger.error("Arg Error: --embedding_output")
            exit(-1)
        trainer.run_node_embedding(args.checkpoint, args.node_count, args.embedding_output)


if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    task = ['train', 'eval', 'recommend', 'node_embedding']
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', dest='task', required=True, choices=task, type=str)
    parser.add_argument('--node_count', dest='node_count', required=True, type=int)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', required=True, type=str)

    parser.add_argument('--l2', dest='l2', required=False, type=float)
    parser.add_argument('--lr', dest='lr', required=False, type=float)
    parser.add_argument('--gru_step', dest='gru_step', required=False, type=int)
    parser.add_argument('--batch_size', dest='batch_size', required=False, type=int, default=512)
    parser.add_argument('--hidden_size', dest='hidden_size', required=False, type=int)
    parser.add_argument('--epochs', dest='epochs', required=False, type=int)
    parser.add_argument('--batch_logging_step', dest='batch_logging_step', required=False, type=int)
    parser.add_argument('--save_step', dest='save_step', required=False, type=int)
    parser.add_argument('--max_test_batch', dest='max_test_batch', required=False, type=int)
    parser.add_argument('--lr_dc', dest='lr_dc', type=int, required=False)
    parser.add_argument('--dc_rate', dest='dc_rate', type=float, required=False)
    parser.add_argument('--early_stop_epochs', dest='early_stop_epochs', required=False, type=int)
    parser.add_argument('--sigma', dest='sigma', required=False, type=float)
    parser.add_argument('--max_len', dest='max_len', required=False, type=int)
    parser.add_argument('--has_uid', dest='has_uid', required=False, type=str2bool)
    parser.add_argument('--feature_init', dest='feature_init', required=False, type=str)
    parser.add_argument('--node_weight', dest='node_weight', required=False, type=str)
    parser.add_argument('--node_weight_trainable', dest='node_weight_trainable', required=False, type=str2bool)
    parser.add_argument('--sq_max_len', dest='sq_max_len', required=False, type=int)

    parser.add_argument('--train_input', dest='train_input', required=False, type=str)
    parser.add_argument('--test_input', dest='test_input', required=False, type=str)
    parser.add_argument('--eval_input', dest='eval_input', required=False, type=str)
    parser.add_argument('--session_input', dest='session_input', required=False, type=str)
    parser.add_argument('--item_lookup', dest='item_lookup', required=False, type=str)
    parser.add_argument('--item_feature', dest='item_feature', required=False, type=str)
    parser.add_argument('--recommend_output', dest='recommend_output', required=False, type=str)
    parser.add_argument('--embedding_output', dest='embedding_output', required=False, type=str)
    parser.add_argument('--rec_extra_count', dest='rec_extra_count', required=False, type=int)
    parser.add_argument('--rec_count', dest='rec_count', required=False, type=int)
    parser.add_argument('--remove_duplicates', dest='remove_duplicates', required=False, type=str2bool)

    args = parser.parse_args()
    main(args)
