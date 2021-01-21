# coding=utf-8
'''raw data on disk to three List[InputExample]'''

import csv
import json
import os
from typing import Dict
from typing import List


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels


class DataProcessor(object):
    def __init__(self, data_dir, example_form=None):
        self.data_dir = data_dir
        self.example_form = example_form

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    # 重要，raw text to BIOS text
    @classmethod
    def _read_json(self, input_file) -> List[Dict]:
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                lines.append({"words": words, "labels": labels})
        return lines


class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(self._read_json(os.path.join(self.data_dir, "train.json")), "train")
        # result = result[:32]

        return result

    def get_dev_examples(self):
        """See base class."""
        result = self._create_examples(self._read_json(os.path.join(self.data_dir, "dev.json")), "dev")
        # result = result[:32]

        return result

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(self.data_dir, "test.json")), "test")

    def get_labels2(self):
        # 中文单字符实体，标为S
        return ['X', 'O',
                'B-address', 'B-book', 'B-company', 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position', 'B-scene',
                'I-address', 'I-book', 'I-company', 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position', 'I-scene',
                'S-address', 'S-book', 'S-company', 'S-game', 'S-government', 'S-movie', 'S-name',
                'S-organization', 'S-position', 'S-scene',
                '<START>', '<END>']

    def get_labels(self):
        if self.example_form == 'span':
            return ['O',
                    'address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization',
                    'position', 'scene']
        else:
            # 中文单字符实体，标为S
            return ['X', 'O',
                    'B-address', 'B-book', 'B-company', 'B-game', 'B-government', 'B-movie', 'B-name',
                    'B-organization', 'B-position', 'B-scene',
                    'I-address', 'I-book', 'I-company', 'I-game', 'I-government', 'I-movie', 'I-name',
                    'I-organization', 'I-position', 'I-scene',
                    'S-address', 'S-book', 'S-company', 'S-game', 'S-government', 'S-movie', 'S-name',
                    'S-organization', 'S-position', 'S-scene']

    def _create_examples(self, lines, set_type) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []

        if self.example_form == 'span':
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line['words']  # char list
                labels = line['labels']  # BIOS list
                subject = get_entities(labels, id2label=None, markup='bios')
                examples.append(InputExample(guid=guid, text_a=text_a, labels=subject))
        else:
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line['words']  # char list
                labels = line['labels']  # BIOS list
                examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


# for span example
def get_entities(seq, id2label, markup='bios'):
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []  # result
    chunk = [-1, -1, -1]  # 一个新chunk (chunk_type, chunk_start, chunk_end)
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)

            chunk = [-1, -1, -1]  # 一个新chunk
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)  # S entity

            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)

            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]

        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


ner_data_processors = {
    'cluener': CluenerProcessor,
    'cner': 'pass'
}

if __name__ == '__main__':
    seq = ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'O', 'S-LOC']
    print(get_entity_bios(seq, id2label=None))
