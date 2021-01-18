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
from nets.plm import MODEL_CLASSES

'''pipeline'''
logger = logging.getLogger(__name__)


def train(args, train_dataset, model):
    """Train the model on `steps` batches"""
    logger.debug('start')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    print('args.train_batch_size')
    print(args.train_batch_size)

    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # scheduler.step()
    lr = 0.0001
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.8)

    global_step = 0
    for epoch in range(int(args.num_train_epochs)):

        for step, batch_data in enumerate(train_dataloader):
            # set model to training mode
            model.train()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            optimizer.zero_grad()

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids, labels=batch_label_ids)

            loss, scores = outputs[:2]

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

    from sklearn.metrics import classification_report

    true_labels = np.array([])
    pred_labels = np.array([])

    with torch.no_grad():
        for batch_data in tqdm(eval_dataloader, desc='dev'):
            model.eval()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids, labels=batch_label_ids)

            # logits: (batch_size, max_len, num_labels)
            loss, logits = outputs[:2]

            # softmax, 最里层dim归一化, shape不变, (batch_size, max_len, num_labels)
            # argmax, 最里层dim取最大值,得到 index对应label, (batch_size, max_len)
            predictions = logits.softmax(dim=-1).argmax(dim=2)

            pred_labels = np.append(pred_labels, predictions.detach().cpu().numpy())
            true_labels = np.append(true_labels, batch_label_ids.detach().cpu().numpy())

    # 查看各个类别的准召
    tags = list(range(args.num_labels))
    print(classification_report(true_labels, pred_labels, labels=tags))


def predict(args, test_dataset, model):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

    results = []
    # fixme
    from data_processor.data_example import get_entity

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(test_dataloader, desc='test')):
            model.eval()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids, labels=batch_label_ids)

            # logits: (batch_size, max_len, num_labels)
            loss, logits = outputs[:2]

            # softmax, 最里层dim归一化, shape不变, (batch_size, max_len, num_labels)
            # argmax, 最里层dim取最大值,得到 index对应label, (batch_size, max_len)
            predictions = logits.softmax(dim=-1).argmax(dim=2)

            predictions = predictions.detach().cpu().numpy().tolist()
            predictions = predictions[0][1:-1]  # [CLS]XXXX[SEP]

            label_entities = get_entity(predictions, args.id2label)
            d = {}
            d['id'] = step
            # d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
            d['entities'] = label_entities
            results.append(d)

    with open('predict_tmp.json', 'w') as writer:
        for d in results:
            writer.write(json.dumps(d) + '\n')


def main(args):
    # 1. setup CUDA, GPU & distributed training
    # cpu
    # if args.local_rank == -1:
    #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # # gpus
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     # torch.distributed.init_process_group(backend="nccl")
    #     args.n_gpu = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 2. processor 初始化
    data_dir = args.data_dir
    task_name = args.task_name
    ner_data_processor = ner_data_processors[task_name](data_dir)

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
    model = model_class.from_pretrained(model_path, num_labels=args.num_labels)
    print(model)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    print("args.device")
    print(args.device)

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

    if args.do_test:
        print('test')
        test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='test')

        model.load_state_dict(torch.load(args.modelfile_finetuned, map_location=lambda storage, loc: storage))
        model.to(args.device)
        predict(args, test_dataset, model)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = (
    #         model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_vocabulary(args.output_dir)
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


import json


def submit(args):
    if args.task_name == 'cluener':

        submit_d = []

        # json to list
        with open('predict_tmp.json', 'r') as fr:
            for line in fr:
                submit_d.append(json.loads(line))

        test_text = []
        with open(os.path.join(args.data_dir, "test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))

        # test_submit = []
        for x, y in zip(submit_d, test_text):
            x['label'] = {}
            if x['entities']:
                for subject in x['entities']:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]

                    if tag not in x['label']:
                        x['label'][tag] = {}
                    word = y['text'][int(start):int(end) + 1]

                    if word not in x['label'][tag]:
                        x['label'][tag][word] = [[start, end]]
                    else:
                        x['label'][tag][word].append([start, end])

            x.pop('entities')
            print(x)

        with open('cluener_predict_me.json', 'w') as writer:
            for line in submit_d:
                writer.write(json.dumps(line, ensure_ascii=False) + '\n')

        # for x, y in zip(test_text, results):
        #     json_d = {}
        #     json_d['id'] = x['id']
        #     json_d['label'] = {}
        #     entities = y['entities']
        #     words = list(x['text'])
        #     if len(entities) != 0:
        #         for subject in entities:
        #             tag = subject[0]
        #             start = subject[1]
        #             end = subject[2]
        #             word = "".join(words[start:end + 1])
        #             if tag in json_d['label']:
        #                 if word in json_d['label'][tag]:
        #                     json_d['label'][tag][word].append([start, end])
        #                 else:
        #                     json_d['label'][tag][word] = [[start, end]]
        #             else:
        #                 json_d['label'][tag] = {}
        #                 json_d['label'][tag][word] = [[start, end]]
        #     test_submit.append(json_d)


class Args(object):
    def __init__(self):
        self.task_name = 'cluener'
        self.data_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'data', 'cluener_public'])
        self.overwrite_cache = 1
        self.local_rank = 0
        self.n_gpu = torch.cuda.device_count()
        self.train_max_seq_length = 55
        self.eval_max_seq_length = 55
        self.model_type = 'bert'

        self.do_train = 0
        self.per_gpu_train_batch_size = 16
        self.num_train_epochs = 3
        self.max_steps = -1
        self.gradient_accumulation_steps = 1

        self.do_eval = 1
        self.eval_batch_size = 16

        self.do_test = 0
        self.test_batch_size = 1


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    args = Args()
    main(args)

    # submit(args)
