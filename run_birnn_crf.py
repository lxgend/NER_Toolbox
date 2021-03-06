# coding=utf-8
import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.optim import SGD

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

        self.weight_decay = 0.01
        self.bert_lr = 0.0001
        self.crf_lr = self.bert_lr * 100

        self.do_train = 1
        self.per_gpu_train_batch_size = 16
        self.num_train_epochs = 3
        self.max_steps = -1
        self.gradient_accumulation_steps = 1

        self.do_eval = 1
        self.eval_batch_size = 16

        self.do_test = 0
        self.test_batch_size = 1

        self.embedding_pretrained = None


class RNN_config(object):
    def __init__(self, device, embedding_pretrained, embedding_dim, vocab_size, num_classes):
        self.device = device
        self.embedding_pretrained = embedding_pretrained
        self.embedding_dim = embedding_dim  # wv 维度

        self.hidden_dim = 128  # 单向lstm hidden_dim
        self.num_rnn_layers = 1
        self.num_directions = 2
        self.dropout = 0.1

        self.vocab_size = vocab_size  # 词表大小
        self.num_classes = num_classes  # label数
        self.max_len = 55  # 单个句子的长度

        self.lr = 1e-3
        self.batch_size = 16
        self.epochs = 10


def train(args, train_dataset, model):
    """Train the model on `steps` batches"""
    logger.debug('start')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    print('args.train_batch_size: %d' % args.train_batch_size)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.8)

    # Prepare optimizer and schedule (linear warmup and decay)
    # 不需要权重衰减的参数
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # crf_param_optimizer = list(model.crf.named_parameters())
    #
    # if args.embedding_pretrained == 'bert':
    #     bert_param_optimizer = list(model.birnn.bert_encoder.named_parameters())
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
    #          'weight_decay': args.weight_decay, 'lr': args.bert_lr},
    #         {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
    #          'lr': args.bert_lr},
    #
    #         {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
    #          'weight_decay': args.weight_decay, 'lr': args.crf_lr},
    #         {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
    #          'lr': args.crf_lr},
    #     ]
    # else:
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
    #          'weight_decay': args.weight_decay, 'lr': args.crf_lr},
    #         {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
    #          'lr': args.crf_lr},
    #     ]
    # from torch.optim import AdamW
    # optimizer = AdamW(optimizer_grouped_parameters)

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

    torch.save(model.state_dict(), args.modelfile_finetuned)

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

            predictions = model.predict(batch_input_ids, batch_input_mask)

            # padding: X
            predictions = list(map(lambda x: x + [0] * (args.eval_max_seq_length - len(x)), predictions))

            predictions = np.array(predictions)

            pred_labels = np.append(pred_labels, predictions)
            true_labels = np.append(true_labels, batch_label_ids.detach().cpu().numpy())

        # 查看各个类别的准召
        tags = list(range(args.num_labels))
        print(classification_report(true_labels, pred_labels, labels=tags))


def main(args):
    # 1. setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 2. data 初始化
    data_dir = args.data_dir
    task_name = args.task_name
    ner_data_processor = ner_data_processors[task_name](data_dir)

    label_list = ner_data_processor.get_labels()
    args.num_labels = len(label_list)
    print("num_labels: %d" % args.num_labels)
    args.id2label = {i: label for i, label in enumerate(label_list)}
    # label2id = {label: i for i, label in enumerate(label_list)}

    model_class, tokenizer_class, model_path = MODEL_CLASSES['bert']

    tokenizer = tokenizer_class.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    print(vocab_size)

    # vocab = tokenizer.get_vocab()

    config = RNN_config(device=device, embedding_pretrained=args.embedding_pretrained, embedding_dim=100,
                        vocab_size=vocab_size,
                        num_classes=args.num_labels)

    model = BiRNN_CRF(config=config)
    model.to(args.device)
    print(model)

    args.modelfile_finetuned = 'finetuned_%s_%s_%s_%s.pt' % (
        args.task_name, config.embedding_pretrained, args.model_type, 'crf')

    if args.do_train:
        # 读数据
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='train')
        print('train_dataset len: %d' % len(train_dataset))

        # train
        global_step = train(args, train_dataset, model)
        print("global_step = %s" % global_step)

    # Evaluation
    if args.do_eval:
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='dev')

        model.load_state_dict(torch.load(args.modelfile_finetuned, map_location=lambda storage, loc: storage))
        model.to(args.device)
        evaluate(args, eval_dataset, model)


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    args = Args()
    main(args)
