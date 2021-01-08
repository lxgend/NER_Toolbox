# coding=utf-8
import torch
import torch.nn as nn
from torchcrf import CRF

# "[START]", "[END]"
START_TAG = 'START'
STOP_TAG = 'STOP'
START_TAG = '<START>'
STOP_TAG = '<STOP>'


class BiLSTM_CRF(nn.Module):

    def __init__(
            self,
            vocab_size,
            batch_size,
            tag_map,
            embedding_dim=100,
            hidden_dim=128,
            dropout=1.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.tag_map = tag_map
        self.tag_size = len(tag_map)
        self.num_rnn_layers = 1

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=self.num_rnn_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)

        # 转移矩阵，随机初始化
        # Matrix of transition parameters. Entry i,j is the score of transitioning *to* i *from* j
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        # self.transitions.data[:, self.tag_map[START_TAG]] = -1000.
        # self.transitions.data[self.tag_map[STOP_TAG], :] = -1000.

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

        self.crf = CRF(self.tag_size, batch_first=True)

    def init_hidden(self):
        """
        random initialize hidden variable 随机初始化
        num_layers * num_directions
        :return: 两个tensor: h0, c0
        """
        h0 = torch.randn(self.num_rnn_layers * 2, self.batch_size, self.hidden_dim // 2)
        c0 = torch.randn(self.num_rnn_layers * 2, self.batch_size, self.hidden_dim // 2)

        return h0, c0

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence, attention_mask=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            attention_mask, 区分是不是padded token
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''

        # sentence.shape: torch.Size([16, 42]), 分别是: batch大小，seq长度
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)

        print(batch_size)
        print(seq_length)

        # 获取bert的词嵌入
        embeds, _ = self.word_embeds(sentence, attention_mask=attention_mask)

        # hidden 初始化
        hidden = self.init_hidden(batch_size)

        # to device
        if embeds.is_cuda:
            hidden = (i.cuda() for i in hidden)

        # lstm 成型, input: bert wv, hidden layer
        # 得到lstm feats,  lstm_out.shape: torch.Size([16, 42, 1000])
        lstm_out, hidden = self.lstm(embeds, hidden)

        # lstm_out.shape: torch.Size([672, 1000])
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        lstm_out = lstm_out.reshape(-1, self.hidden_dim * 2)

        # 通过dropout1层
        # d_lstm_out.shape: torch.Size([672, 1000])
        d_lstm_out = self.dropout1(lstm_out)

        # 通过liner层
        #  l_out.shape: torch.Size([672, 4])
        l_out = self.liner(d_lstm_out)

        # reshape
        # lstm_feats.shape: torch.Size([16, 42, 4])
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)

        return lstm_feats

        # def my_neg_log_likelihood_loss_func(self, sentence, tags):
        #     feats = self._get_lstm_features(sentence)
        #     forward_score = self._forward_alg(feats)
        #     gold_score = self._score_sentence(feats, tags)
        #     return forward_score - gold_score

    # 构建了自己model的loss func, 传回作为bpp
    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        # 调用 crf里的loss func
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)

        # 16
        batch_size = feats.size(0)

        # 得到batch中单个exp的loss
        loss_value /= float(batch_size)
        return loss_value


if __name__ == '__main__':
    pass