# coding=utf-8
import logging
import os

import torch
from transformers import AlbertForTokenClassification
from transformers import BertForTokenClassification
from transformers import BertTokenizer

from data_processor.data_example import ner_data_processors
from data_processor.dataset_utils import load_and_cache_examples
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
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    # gpus
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # 2. processor 初始化
    data_dir = args.data_dir
    task_name = args.task_name
    ner_data_processor = ner_data_processors[task_name](data_dir)

    label_list = ner_data_processor.get_labels()
    num_labels = len(label_list)

    # id2label = {i: label for i, label in enumerate(label_list)}
    # label2id = {label: i for i, label in enumerate(label_list)}

    # 3. Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # PATH_MODEL_BERT = '/home/ubuntu/MyFiles/roberta_wwm_ext_zh_hit_pt'
    PATH_MODEL_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/albert_zh_xxlarge_google_pt'
    tokenizer = tokenizer_class.from_pretrained(PATH_MODEL_BERT)
    model = model_class.from_pretrained(PATH_MODEL_BERT, num_labels=num_labels)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    # 4.分支操作
    # Training
    if args.do_train:
        # 读数据
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='dev')
        print(train_dataset)

        # train
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

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


class Args(object):
    def __init__(self):
        self.task_name = 'cluener'
        self.data_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'data', 'cluener_public'])
        self.overwrite_cache = 1
        self.local_rank = -1
        self.no_cuda = 1
        self.n_gpu = torch.cuda.device_count()
        self.train_max_seq_length = 55
        self.eval_max_seq_length = 55
        self.model_type = 'bert'

        self.do_train = 1
        self.per_gpu_train_batch_size = 8
        self.num_train_epochs = 3
        self.max_steps = -1
        self.gradient_accumulation_steps = 1


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    args = Args()
    main(args)
