# coding=utf-8
import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, DistributedSampler
from torch.utils.data import SequentialSampler
from transformers import BertTokenizer

from data_processor.data_example import ner_data_processors
from data_processor.dataset_utils import load_and_cache_examples
logger = logging.getLogger(__name__)
from nets.birnn import BiRNN

class Args(object):
    def __init__(self):
        self.task_name = 'cluener'
        self.data_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'data', 'cluener_public'])
        self.overwrite_cache = 1
        self.local_rank = -1
        self.n_gpu = torch.cuda.device_count()


        self.train_max_seq_length = 55
        self.eval_max_seq_length = 55
        self.model_type = 'lstm'

        self.do_train = 1
        self.per_gpu_train_batch_size = 16
        self.num_train_epochs = 3
        self.max_steps = -1
        self.gradient_accumulation_steps = 1

        self.do_eval = 0
        self.eval_batch_size = 16

        self.do_test = 0
        self.test_batch_size = 1


class RNN_config(object):
    def __init__(self, embedding_dim, vocab_size, num_classes):
        self.embedding_pretrained = None
        self.embedding_dim = embedding_dim  # wv 维度
        self.hidden_dim = 64

        self.num_rnn_layers = 1
        self.num_directions = 2
        self.dropout = 0.1

        self.vocab_size = vocab_size  # 词表大小
        self.num_classes = num_classes  # label数
        self.max_len = 64  # 单个句子的长度
        self.lr = 1e-3
        self.batch_size = 16
        self.epochs = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args, train_dataset, model):
    """Train the model on `steps` batches"""
    logger.debug('start')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    print('args.train_batch_size: %d' % args.train_batch_size)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    for epoch in range(int(args.num_train_epochs)):
        for step, batch_data in enumerate(train_dataloader):
            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            model(batch_input_ids, batch_input_mask)


def main(args):
    # 1. setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 2. data 初始化
    data_dir = args.data_dir
    task_name = args.task_name
    ner_data_processor = ner_data_processors[task_name](data_dir)


    label_list = ner_data_processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}

    num_labels = len(label_list)
    print("num_labels: %d" % num_labels)

    PATH_MODEL_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/roberta_wwm_ext_zh_hit_pt'
    tokenizer = BertTokenizer.from_pretrained(PATH_MODEL_BERT)

    vocab_size = tokenizer.vocab_size

    print(label2id)
    print(vocab_size)

    vocab = tokenizer.get_vocab()

    config = RNN_config(embedding_dim=100, vocab_size=vocab_size, num_classes=num_labels)
    model = BiRNN(config=config)
    model.to(args.device)
    print(model)

    if args.do_train:
        # 读数据
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='train')
        print('train_dataset len: %d' % len(train_dataset))

        train(args, train_dataset, model)

        # train
        # global_step = train(args, train_dataset, model)
        # print("global_step = %s" % global_step)

    '''

    # Evaluation
    if args.do_eval:
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='dev')

        model.load_state_dict(torch.load('cluener_fine_tuned.pt', map_location=lambda storage, loc: storage))
        model.to(args.device)
        evaluate(args, eval_dataset, model)
        
    '''


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    args = Args()
    main(args)
