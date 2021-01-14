# coding=utf-8
import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.optim import SGD
from transformers import BertTokenizer

from data_processor.data_example import ner_data_processors
from data_processor.dataset_utils import load_and_cache_examples

logger = logging.getLogger(__name__)
from nets.birnn import BiRNN_CRF
from nets.plm import MODEL_CLASSES

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

        self.do_eval = 1
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

    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.8)

    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, batch_data in enumerate(train_dataloader):
            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            loss = model(batch_input_ids, batch_input_mask, batch_label_ids)

            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                print('epoch: {} | step: {} | loss: {}'.format(epoch, step, loss.item()))

            global_step += 1

    torch.save(model.state_dict(), 'cluener_fine_tuned_lstmcrf.pt')

    return global_step

def evaluate(args, eval_dataset, model):
    from sklearn.metrics import classification_report
    import numpy as np
    from tqdm import tqdm


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    true_labels = np.array([])
    pred_labels = np.array([])

    with torch.no_grad():
        for batch_data in tqdm(eval_dataloader, desc='dev'):
            model.eval()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            print(batch_label_ids)

            predictions = model.predict(batch_input_ids, batch_input_mask)

            # padding
            predictions = list(map(lambda x: x + [31] * (55 - len(x)), predictions))
            predictions = np.array(predictions)

            pred_labels = np.append(pred_labels, predictions)
            true_labels = np.append(true_labels, batch_label_ids.detach().cpu().numpy())

        # 查看各个类别的准召
        tags = list(range(34))
        print(classification_report(pred_labels, true_labels, labels=tags))


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

    model_class, tokenizer_class, model_path = MODEL_CLASSES['bert']

    tokenizer = tokenizer_class.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size

    print(label2id)
    print(vocab_size)

    # vocab = tokenizer.get_vocab()

    config = RNN_config(embedding_dim=100, vocab_size=vocab_size, num_classes=num_labels)
    model = BiRNN_CRF(config=config)
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

    # Evaluation
    if args.do_eval:
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='dev')

        model.load_state_dict(torch.load('cluener_fine_tuned_lstmcrf.pt', map_location=lambda storage, loc: storage))
        model.to(args.device)
        evaluate(args, eval_dataset, model)


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    args = Args()
    main(args)
