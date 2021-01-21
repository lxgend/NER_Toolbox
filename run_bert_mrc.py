# coding=utf-8
import logging
import os

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from data_processor.data_example import ner_data_processors
from data_processor.dataset_utils import load_and_cache_examples
from nets.bert_mrc import Bert_MRC
from nets.plm import MODEL_CLASSES

'''pipeline'''
logger = logging.getLogger(__name__)


def train(args, train_dataset, model):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    print('train_batch_size %d' % args.train_batch_size)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    lr = 0.0001
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.8)

    global_step = 0
    for epoch in range(int(args.num_train_epochs)):

        for step, batch_data in enumerate(train_dataloader):
            # set model to training mode
            model.train()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_start_ids, batch_end_ids = batch_data
            optimizer.zero_grad()
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids, start_positions=batch_start_ids,
                            end_positions=batch_end_ids)

            loss = outputs[0]
            # print(loss)

            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                print('epoch: {} | step: {} | loss: {}'.format(epoch, step, loss.item()))

            global_step += 1

    torch.save(model.state_dict(), args.modelfile_finetuned)
    return global_step


def evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    true_labels = np.array([])
    pred_labels = np.array([])

    with torch.no_grad():
        for batch_data in tqdm(eval_dataloader, desc='dev'):
            model.eval()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_star_ids, batch_end_ids= batch_data


def main(args):
    # 1. setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.device)

    # 2. processor 初始化
    ner_data_processor = ner_data_processors[args.task_name](args.data_dir)

    label_list = ner_data_processor.get_labels()
    args.num_labels = len(label_list)
    print("num_labels: %d" % args.num_labels)
    args.id2label = {i: label for i, label in enumerate(label_list)}
    # label2id = {label: i for i, label in enumerate(label_list)}

    # 3. Load pretrained model and tokenizer
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_class, tokenizer_class, model_path = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(model_path)

    model = Bert_MRC(path=model_path, num_tag=args.num_labels)
    print(model)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    args.modelfile_finetuned = 'finetuned_%s_%s.pt' % (args.task_name, args.model_type)

    # 4.分支操作
    # Training
    if args.do_train:
        # 读数据
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='train')
        print('train_dataset')
        print(len(train_dataset))

        # train
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        global_step = train(args, train_dataset, model)
        print("global_step = %s" % global_step)

    # Evaluation
    results = {}
    if args.do_eval:
        print('evaluate')

        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='dev')

        model.load_state_dict(torch.load(args.modelfile_finetuned, map_location=lambda storage, loc: storage))
        model.to(args.device)
        evaluate(args, eval_dataset, model)


class Args(object):
    def __init__(self):
        self.task_name = 'cluener'
        self.data_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'data', 'cluener_public'])
        self.overwrite_cache = 1
        self.local_rank = 0
        self.n_gpu = torch.cuda.device_count()

        self.model_type = 'bert'
        self.train_max_seq_length = 55
        self.eval_max_seq_length = 55

        self.do_train = 1
        self.per_gpu_train_batch_size = 16
        self.num_train_epochs = 3
        self.max_steps = -1
        self.gradient_accumulation_steps = 1

        self.do_eval = 1
        self.eval_batch_size = 16

        self.do_test = 0
        self.test_batch_size = 1


if __name__ == '__main__':
    args = Args()
    main(args)
