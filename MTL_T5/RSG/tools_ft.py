import re
import os
import random
import evaluate
import numpy as np
from torch import manual_seed
from num2words import num2words

from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
import torch


accuracy = evaluate.load("accuracy")

def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = np.argmax(predictions[0], axis=1)
    else:
        predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def compute_mc_accuracy(eval_pred):
    predictions, labels = eval_pred
    # print(print(predictions))
    if isinstance(predictions, tuple):
        predictions = np.argmax(predictions[0], axis=1)
    else:
        predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def process_sentence(sentence: str) -> str:
    res = sentence.lower()
    res = res.replace('.', '')
    res = res.replace(',', '')
    res = res.replace('?', '')
    res = res.replace('!', '')
    res = res.replace('"', '')
    res = res.replace("'", "")
    res = res.replace("№", "")
    res = res.replace("—", "")
    res = res.replace("-", "")
    res = res.replace("(", "")
    res = res.replace(")", "")
    res = res.replace("$", "")
    res = res.replace("–", "")
    res = res.replace("«", "")
    res = res.replace("»", "")
    res = res.replace('-', ' ')
    res = res.replace("“", "")
    res = res.replace("”", "")

    for i in re.findall(r'\d+', res):
        res = res.replace(i,  num2words(i, lang='ru'))
    res = res.replace("  ", " ")
    return res.strip()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    manual_seed(seed)


def preprocess_rcb(row, tokenizer):
    sentence = tokenizer(process_sentence(row['premise']), process_sentence(row['hypothesis']))
    label_map = {'contradiction': [1., 0., 0.] , 'entailment': [0., 1., 0.], 'neutral': [0., 0., 1.]}
    if 'label' in row.keys():
        sentence['label'] = label_map[row['label']]
    return sentence


def prepreprocess_parus(row):
    qmap = {'cause': 'причина', 'effect': 'эффект'}
    res = {
        'premise': process_sentence(row['premise']),
        'choice1': process_sentence(row['choice1']),
        'choice2': process_sentence(row['choice2']),
        'question': qmap[row['question']],
    }
    return res


def preprocess_parus(examples, tokenizer):
    ending_names = ["choice1", "choice2"]
    first_sentences = [[context] * len(ending_names) for context in examples["premise"]]
    question_headers = examples["question"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}


def preprocess_terra(row, tokenizer):
    sentence = tokenizer(process_sentence(row['premise']), process_sentence(row['hypothesis']))
    label_map = {'entailment': 0, 'not_entailment': 1}
    if 'label' in row.keys():
        sentence['label'] = label_map[row['label']]
    return sentence


def preprocess_lidirus(row, tokenizer):
    sentence = tokenizer(process_sentence(row['sentence1']), process_sentence(row['sentence2']))
    label_map = {'entailment': 0, 'not_entailment': 1}
    if 'label' in row.keys():
        sentence['label'] = label_map[row['label']]
    return sentence


def preprocess_dnqa(row, tokenizer):
    sentence = tokenizer(process_sentence(row['question']), process_sentence(row['passage']), max_length=1023)
    if 'blabel' in row.keys():
        sentence['label'] = int(1 if row['blabel'] else 0)
    return sentence


def preprocess_rwsd(row, tokenizer, sep_token: str = '</s>'):
    sentence = tokenizer(
        process_sentence(row['text']),
        process_sentence(row['target']['span1_text']) + sep_token + process_sentence(row['target']['span2_text'])
    )
    if 'label' in row.keys():
        sentence['ilabel'] = int(1 if row['label'] else 0)
    return sentence


def preprocess_russe(row, tokenizer, sep_token: str = '</s>'):
    sentence = tokenizer(
        process_sentence(row['sentence1']),
        process_sentence(row['sentence2']) + sep_token + process_sentence(row['word'])
    )
    if 'label' in row.keys():
        sentence['ilabel'] = int(1 if row['label'] else 0)
    return sentence


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # label_name = "label" if "label" in features[0].keys() else "labels"
        label_name = 'label'
        # print(features)
        label_check = False
        if label_name in features[0].keys():
            label_check = True
            labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if label_check:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
