# coding=utf-8
import json
import logging

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer):
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

            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                print('epoch: {} | step: {} | loss: {}'.format(epoch, step, loss.item()))

            global_step += 1

    torch.save(model.state_dict(), 'cluener_fine_tuned.pt')

    return global_step


def evaluate(args, eval_dataset, model, tokenizer, prefix=""):
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
    tags = list(range(34))
    print(classification_report(pred_labels, true_labels, labels=tags))


def predict(args, test_dataset, model, tokenizer, prefix=""):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

    results = []

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


if __name__ == '__main__':
    pass
