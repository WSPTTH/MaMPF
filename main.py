import os
import preprocess
import Markov
import argparse


home = os.getcwd()
record_dir = os.path.join(home, 'record')
data_dir = os.path.join(home, 'filter')
pred_dir = os.path.join(home, 'result')

for dirx in [record_dir, data_dir, pred_dir]:
    if not os.path.exists(dirx):
        os.makedirs(dirx)

train_record = os.path.join(record_dir, 'train.json')
test_record = os.path.join(record_dir, 'test.json')
status_label = os.path.join(data_dir, 'status.label')

def main():
    parser = argparse.ArgumentParser('run the markov model')
    parser.add_argument('--mode', type=str, choices=['prepro', 'markov'])
    parser.add_argument('--train_json', type=str, default=train_record, help='the processed train json file')
    parser.add_argument('--test_json', type=str, default=test_record, help='the processed test json file')
    parser.add_argument('--status_label', type=str, default=status_label, help='the status label')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='where to read data')
    parser.add_argument('--pred_dir', type=str, default=pred_dir, help='the dir to save predict result')

    # for preprocessing
    parser.add_argument('--class_num', type=int, default=18, help='the class number')
    parser.add_argument('--min_length', type=int, default=2, help='the flow under this parameter will be filtered')
    parser.add_argument('--max_packet_length', type=int, default=6000, help='the largest packet length')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='ratio of train set of target app')
    parser.add_argument('--keep_ratio', type=float, default=1.0,
                        help='ratio of keeping the example (for small dataset test)')

    parser.add_argument('--markov_models', type=str, default='SLC-LR', help='markov methods, split by \'#\'')

    config = parser.parse_args()
    if config.mode == 'prepro':
        preprocess.preprocess(config)
    elif config.mode == 'markov':
        Markov.markov(config)
    else:
        print('unknown mode, only support train now')
        raise Exception


if __name__ == '__main__':
    main()
