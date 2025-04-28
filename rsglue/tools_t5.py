import re
import os
import random
import evaluate
import numpy as np
from torch import manual_seed
from num2words import num2words
import datasets
from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
import torch
from sklearn.metrics import accuracy_score


# accuracy = evaluate.load("accuracy")
# accuracy = accuracy_score


class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }
        
accuracy = Accuracy()
        

def compute_mc_accuracy(eval_pred):
    preds, labels = eval_pred          # Trainer отдаёт (predictions, labels)

    # --- обработка logits ---
    if isinstance(preds, tuple):       # иногда Trainer кладёт их в кортеж
        preds = preds[0]
    if preds.ndim > 1:                 # [batch, num_labels] → индексы классов
        preds = np.argmax(preds, axis=1)

    # --- обработка меток ---
    if labels.ndim > 1:                # one‑hot? → превращаем в индексы
        labels = np.argmax(labels, axis=1)

    return accuracy.compute(
        predictions=preds.tolist(),
        references=labels.tolist()
    )





def compute_accuracy(eval_pred):
    logits, labels = eval_pred            # Trainer всегда даёт пару

    # logits может быть tuple (outputs.logits,) — разворачиваем
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=1)     # argmax по классу
    return accuracy.compute(
        predictions=preds.tolist(),       # evaluate ждёт list/np.ndarray
        references=labels.tolist()
    )

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
    encoding = tokenizer(
        process_sentence(row['premise']),
        process_sentence(row['hypothesis']),
        truncation=True,
        padding='max_length',
        max_length=512,
    )
    label_map = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    if 'label' in row:
        encoding['label'] = label_map[row['label']]
    return encoding


def prepreprocess_parus(row: dict) -> dict | None:
    premise  = process_sentence(row.get("premise", ""))
    choice1  = process_sentence(row.get("choice1", ""))
    choice2  = process_sentence(row.get("choice2", ""))
    question = row.get("question", "").lower()        # рус/англ, регистр

    if question not in ["cause", "effect", "причина", "эффект"]:
        return None
    if not (premise or choice1 or choice2):
        return None

    out = {
        "premise": premise,
        "choice1": choice1,
        "choice2": choice2,
        "question": question,
    }
    if "label" in row:            # оставляем метку для train/val
        out["label"] = row["label"]
    return out


def preprocess_parus_binary(row: dict, tokenizer, max_len=512) -> dict | None:
    text = (
        f"{row['premise']} {tokenizer.eos_token} "
        f"{row['choice1']} {tokenizer.eos_token} "
        f"{row['choice2']} {tokenizer.eos_token}"
    )

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        add_special_tokens=True,
    )
    if len(enc["input_ids"]) == 0:
        return None

    q = row["question"].lower()
    if q in ("cause", "причина"):
        enc["labels"] = 0
    elif q in ("effect", "эффект"):
        enc["labels"] = 1
    elif "label" in row:
        enc["labels"] = int(row["label"])
    else:
        return None

    return enc



def preprocess_terra(row: dict, tokenizer) -> dict:
    premise = process_sentence(row["premise"])
    hypothesis = process_sentence(row["hypothesis"])

    # Жёсткая проверка на пустые значения
    if not premise.strip() or not hypothesis.strip():
        return None

    # Токенизация с ограничением длины
    enc = tokenizer(
        premise,
        hypothesis,
        add_special_tokens=True,  # Разрешаем стандартные спец-токены
        max_length=512,
        truncation=True,
        padding=False
    )

    # Добавляем явный EOS, если его нет
    if enc['input_ids'][-1] != tokenizer.eos_token_id:
        enc['input_ids'].append(tokenizer.eos_token_id)
        enc['attention_mask'].append(1)

    # Жёсткая проверка валидности последовательности
    if len(enc['input_ids']) == 0 or enc['input_ids'][-1] != tokenizer.eos_token_id:
        return None

    # Преобразование метки
    label_map = {"entailment": 0, "not_entailment": 1}
    if "label" in row:
        enc["label"] = label_map.get(row["label"], -1)
        
    if not enc['input_ids']:
        enc['input_ids'] = [tokenizer.pad_token_id]
        enc['attention_mask'] = [0]

    return enc



# ------------------------------------------------------------------
# функция препроцессинга – ОСТАВЛЕНА в том же виде, только добавлена
# гарантия наличия EOS в конце
# ------------------------------------------------------------------
def preprocess_lidirus(example, tokenizer):
    text1 = example["sentence1"]
    text2 = example["sentence2"]

    enc = tokenizer(
        text1,
        text2,
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=True,     # T5 всё-таки не ставит </s> для encoder-input
    )

    # ──────────────────────────────────────────────────────────────
    #  гарантируем, что в последовательности есть EOS-токен
    # ──────────────────────────────────────────────────────────────
    if tokenizer.eos_token_id not in enc["input_ids"]:
        ids = enc["input_ids"]
        mask = enc["attention_mask"]

        if len(ids) == 512:          # уже на пределе длины – заменяем последний токен
            ids[-1]  = tokenizer.eos_token_id
        else:                        # есть запас – просто дописываем
            ids.append(tokenizer.eos_token_id)
            mask.append(1)

        enc["input_ids"]      = ids
        enc["attention_mask"] = mask
    # ──────────────────────────────────────────────────────────────

    # метка → целое, если она присутствует
    if "label" in example:
        enc["labels"] = 0 if example["label"] == "entailment" else 1

    return enc





import torch
import torch.nn.functional as F



# ---------------------- Н О В Ы Й  preprocess_dnqa -----------------------
def preprocess_dnqa(row: dict, tokenizer) -> dict:
    question = process_sentence(row["question"])
    passage  = process_sentence(row["passage"])

    # пропускаем строки с пустыми полями
    if not question or not passage:
        return None

    enc = tokenizer(
        question,
        passage,
        max_length=512,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,          # заставляем токенизатор вставить <s> и </s>
    )

    # ──────────────────────────────────────────────────────────────
    # гарантируем наличие EOS в input_ids
    # ──────────────────────────────────────────────────────────────
    eos_id = tokenizer.eos_token_id
    if eos_id not in enc["input_ids"]:
        ids  = enc["input_ids"]
        mask = enc["attention_mask"]

        if len(ids) == 512:        # замещаем последний токен
            ids[-1] = eos_id
        else:                      # дописываем
            ids.append(eos_id)
            mask.append(1)

        enc["input_ids"]      = ids
        enc["attention_mask"] = mask
    # ──────────────────────────────────────────────────────────────

    # текстовая метка → 0 / 1  **и кладём в ключ 'labels'**
    if "label" in row:
        enc["labels"] = 1 if row["label"] == "true" else 0

    return enc




# ---------- RWSD preprocessing ----------
def preprocess_rwsd(row: dict, tokenizer, sep_token: str = "</s>") -> dict:
    text       = process_sentence(row["text"]) + sep_token
    span1_text = process_sentence(row["target"]["span1_text"]) + sep_token
    span2_text = process_sentence(row["target"]["span2_text"]) + sep_token

    enc = tokenizer(
        text,
        span1_text + span2_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )

    # метка → int  (НЕ bool!)
    if "label" in row:
        # в файлах RWSD метка обычно строка "True"/"False" или bool
        enc["labels"] = 1 if str(row["label"]).lower() in ("true", "1") else 0
        enc["labels"] = int(enc["labels"])        # <─ гарантируем int

    return enc




# def preprocess_russe(row: dict, tokenizer,
#                      sep_token: str = "</s>") -> dict:
#     enc = tokenizer(
#         process_sentence(row["sentence1"]),
#         f"{process_sentence(row['sentence2'])}{sep_token}"
#         f"{process_sentence(row['word'])}",
#         padding="max_length", truncation=True
#     )
#     if "label" in row:
#         enc["label"] = int(bool(row["label"]))    # 0 / 1
#     return enc



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
