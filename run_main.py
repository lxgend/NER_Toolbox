# coding=utf-8
import logging
import os

import torch
from transformers import AlbertForTokenClassification
from transformers import BertForTokenClassification
from transformers import BertTokenizer

from data_processor.data_example import ner_data_processors
from data_processor.dataset_utils import load_and_cache_examples
from model_pipeline import evaluate
from model_pipeline import predict
from model_pipeline import train

'''pipeline'''
logger = logging.getLogger(__name__)

# 模型选择
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertForTokenClassification, BertTokenizer),
    'albert': (AlbertForTokenClassification, BertTokenizer),
    # 'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    # 'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}


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

    num_labels = len(label_list)
    print("num_labels")
    print(num_labels)

    args.id2label = {i: label for i, label in enumerate(label_list)}
    # label2id = {label: i for i, label in enumerate(label_list)}

    # 3. Load pretrained model and tokenizer
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # PATH_MODEL_BERT = '/home/ubuntu/MyFiles/roberta_wwm_ext_zh_hit_pt'
    PATH_MODEL_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/roberta_wwm_ext_zh_hit_pt'
    tokenizer = tokenizer_class.from_pretrained(PATH_MODEL_BERT)
    model = model_class.from_pretrained(PATH_MODEL_BERT, num_labels=num_labels)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    print("args.device")
    print(args.device)

    # 4.分支操作
    # Training
    if args.do_train:
        # 读数据
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='train')
        print('train_dataset')
        print(len(train_dataset))

        # train
        # global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        global_step = train(args, train_dataset, model, tokenizer)
        print("global_step = %s", global_step)

    # Evaluation
    results = {}
    if args.do_eval:
        print('evaluate')

        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='dev')

        model.load_state_dict(torch.load('cluener_fine_tuned.pt', map_location=lambda storage, loc: storage))
        model.to(args.device)
        evaluate(args, eval_dataset, model, tokenizer)

    if args.do_test:
        print('test')
        test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='test')

        model.load_state_dict(torch.load('cluener_fine_tuned.pt', map_location=lambda storage, loc: storage))
        model.to(args.device)
        predict(args, test_dataset, model, tokenizer)

        # preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
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
