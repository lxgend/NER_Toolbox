# coding=utf-8
import torch.nn as nn
from torchcrf import CRF
from transformers import BertForTokenClassification, BertModel

class Bert_CRF(nn.Module):
    def __init__(self,
                 path,
                 num_tag):  # len
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(path, num_labels=num_tag)
        self.crf = CRF(num_tag, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        # 不在此步计算loss，因此不需要输入label
        # 查看一下有label和无，logit是否一致
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        logits = bert_outputs[0]

        # torch.Size([batch_size, seq_len, 34])
        # print(logits)

        log_likelihood = self.crf(emissions=logits, tags=labels, mask=attention_mask, reduction='token_mean')

        outputs = (logits,)

        # negative log likelihood loss
        outputs = (-1 * log_likelihood,) + outputs
        return outputs  # (loss), bert_logits

    # 没有用到
    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss


if __name__ == '__main__':
    print(BertForTokenClassification)
    print(BertModel)
