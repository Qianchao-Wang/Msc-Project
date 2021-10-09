import argparse

import warnings
warnings.filterwarnings("ignore")


#def main(args):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
