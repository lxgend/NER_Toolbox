# coding=utf-8
import torch.nn as nn
from transformers import BertModel

class Bert_MRC(nn.Module):
    def __init__(self,
                 path,
                 hidden_size,
                 hidden_dropout_prob,
                 mrc_dropout,
                 num_tag):  # len
        super().__init__()
        self.bert = BertModel.from_pretrained(path)
        self.num_labels = num_tag
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # self.soft_label = config.soft_label
        # self.loss_type = config.loss_type

        self.start_fc = nn.Linear(hidden_size, num_tag)
        self.end_fc = nn.Linear(hidden_size, num_tag)
        self.span_embedding = MultiNonLinearClassifier(hidden_size * 2, 1, mrc_dropout)
        self.hidden_size = hidden_size

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            start_positions=None,
            end_positions=None
    ):
        # 不在此步计算loss，因此不需要输入label
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        logits = bert_outputs[0]
        logits = self.dropout(logits)
        start_logits = self.start_fc(logits)
        end_logits = self.end_fc(logits)

        # start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        # end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        # start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # # [batch, seq_len, seq_len, hidden]
        # end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # # [batch, seq_len, seq_len, hidden*2]
        # span_matrix = torch.cat([start_extend, end_extend], 3)
        # # [batch, seq_len, seq_len]
        # span_logits = self.span_embedding(span_matrix).squeeze(-1)


        return start_logits, end_logits, span_logits


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None):

        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_fc(sequence_output)

        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type =='lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


if __name__ == '__main__':
    print(BertModel)