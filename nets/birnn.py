# coding=utf-8
import torch
import torch.nn as nn
from torchcrf import CRF

torch.manual_seed(123)  # 保证每次运行初始化的随机数相同

class BiRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.device

        if config.embedding_pretrained is not None:
            # 特指plm
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.vocab_size - 1)

        self.embedding_pretrained = config.embedding_pretrained
        self.hidden_dim = config.hidden_dim
        self.num_rnn_layers = config.num_rnn_layers
        self.num_directions = config.num_directions

        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.hidden_dim,
                            num_layers=config.num_rnn_layers,
                            dropout=0,  # num_layers=1时是无效的
                            bidirectional=(config.num_directions == 2),
                            batch_first=True)

        self.hidden2tag = nn.Linear(config.hidden_dim * config.num_directions * config.num_rnn_layers,
                                    config.num_classes)

        # self.act_func = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(config.dropout)

        # self.vocab_size = config.vocab_size
        # self.batch_size = config.batch_size

    def init_hidden(self, batch_size):
        """
        random initialize hidden variable 随机初始化
        num_layers * num_directions
        hidden_dim 和 self.lstm保持一致
        :return: 两个tensor: h0, c0
        """
        h0 = torch.randn(self.num_rnn_layers * self.num_directions, batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_rnn_layers * self.num_directions, batch_size, self.hidden_dim)
        return h0, c0

    def forward4(self, sentence):
        # sentence.shape: torch.Size([16, 42]), 分别是: batch大小，seq长度
        # 由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)

        embeds = self.embedding(sentence)

        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])

        # lstm最初的前项输出
        h0, c0 = self.init_hidden(batch_size)

        # input [batch_size, seq_len, embeding]=[128, 32, 300]
        # out[seq_len, batch_size, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(sentence, (h0, c0))

        x = self.liner(x)
        x = self.act_func(x)

        return x

    def forward3(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        return y

    def forward2(self, inputs_ids, input_mask):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output = self.layer_norm(seqence_output)
        features = self.classifier(seqence_output)
        return features

    def forward(self, inputs_ids, input_mask):
        # x shape:
        # [batch_size, padded sentence_length]  ids
        # [batch_size, padded sentence_length, embedding_size]   pretrain emb

        batch_size = inputs_ids.size(0)                  # 获取当前数据实际的batch
        # h0, c0 = self.init_hidden(batch_size)   # 初始化 lstm最初的前项输出

        if self.embedding_pretrained is not None:
            pass
        else:
            embeds = self.embedding(inputs_ids)
            embeds = embeds * input_mask.float().unsqueeze(2)

        # out [batch_size, seq_len, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # hn, cn shape: 同 h0, c0
        # out, (hn, cn) = self.lstm(embs, (h0, c0))
        lstm_out, (hn, cn) = self.lstm(embeds)

        lstm_out = self.hidden2tag(lstm_out)

        print(lstm_out.shape)

        return lstm_out

        # x = h_n  # [num_layers*num_directions, batch_size, hidden_size]
        # x = x.permute(1, 0, 2)  # [batch_size, num_layers*num_directions, hidden_size]
        # x = x.contiguous().view(batch_size,
        # self.num_layers * self.num_directions * self.hidden_size)  # [batch_size, num_layers*num_directions*hidden_size]
        # x = self.liner(x)
        # x = self.act_func(x)
        # return x

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()

        # max_len, batchsize, -1
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

class BiRNN_CRF(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.birnn = BiRNN(config)
        self.transitions = nn.Parameter(torch.randn(config.num_classes, config.num_classes))
        self.crf = CRF(config.num_classes, batch_first=True)

    def forward(self, x, mask, y):  # for training
        mask = xw.data.gt(0).float()
        h = self.rnn(xc, xw, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y, mask)
        return Z - score  # NLL loss

    def decode(self, xc, xw):  # for prediction
        mask = xw.data.gt(0).float()
        h = self.rnn(xc, xw, mask)
        return self.crf.decode(h, mask)





class Config(object):
    def __init__(self):
        self.embedding_pretrained = None
        self.embedding_dim = 100  # wv 维度
        self.hidden_dim = 64

        self.num_rnn_layers = 1
        self.num_directions = 2
        self.dropout = 0.1

        self.vocab_size = 5000  # 词表大小

        self.num_classes = 2  # 二分类
        self.max_len = 64  # 单个句子的长度
        self.lr = 1e-3
        self.batch_size = 16
        self.epochs = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':


    # "[START]", "[END]"
    START_TAG = '<START>'
    STOP_TAG = '<STOP>'

    config = Config()

    embed_size, num_hiddens, num_layers = 300, 100, 2
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    # 加载维基百科预训练词向量(使用fasttext),cache为保存目录
    fasttext_vocab = Vocab.FastText(cache=os.path.join(DATA_ROOT, "fasttext"))

    # fastext_vocab是预训练好的词向量
    net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos,
                                                              fasttext_vocab))
    net.embedding.weight.requires_grad = False
    net = net.to(device)

    model = BiLSTM(config)

    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        loss_ = loss(y_hat, y)