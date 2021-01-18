# coding=utf-8
from transformers import BertForTokenClassification
from transformers import BertTokenizer

# PATH_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/roberta_wwm_ext_zh_hit_pt'
PATH_BERT = '/home/dc2-user/modelfile/roberta_wwm_ext_zh_hit_pt'


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertForTokenClassification, BertTokenizer, PATH_BERT),
}