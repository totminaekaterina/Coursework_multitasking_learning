import re
import os
import random
import evaluate
import numpy as np
from torch import manual_seed
from num2words import num2words
import datasets
from sklearn.metrics import accuracy_score
from datasets import Dataset
import pathlib


# -------- rcb --------
def preprocess_rcb_t5(ex: dict, tok, max_len: int = 256):
    premise    = ex["premise"].strip()
    hypothesis = ex["hypothesis"].strip()

    src = (
        f"Следует из PREMISE: '{premise}' \n"
        f"HYPOTHESIS:'{hypothesis}'?\n"
        f"Ответь одним словом: contradiction, entailment, neutral."
    )
    model_inputs = tok(src, truncation=True, max_length=max_len)

    tgt_text = ex["label"] if "label" in ex else ""
    with tok.as_target_tokenizer():
        labels = tok(tgt_text, max_length=8, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# MuSeRC
LABEL_MUSERC = {0: "false", 1: "true"}
def build_muserc_dataset_t5(path: pathlib.Path,
                            tok,
                            max_len: int = 512) -> Dataset:
    rows = []
    for samp in Dataset.from_json(str(path)):
        pid         = int(samp["idx"])
        passage_txt = samp["passage"]["text"]

        for q in samp["passage"]["questions"]:
            question_txt = q["question"]
            q_idx        = int(q["idx"])

            for ans in q["answers"]:
                answer_txt = ans["text"]
                a_idx      = int(ans["idx"])
                label      = ans.get("label", -1)

                prompt = (
                    f"Текст: {passage_txt}{tok.eos_token}"
                    f"Вопрос: {question_txt}{tok.eos_token}"
                    f"Вариант-ответ: {answer_txt}{tok.eos_token}"
                    "Верно ли, что вариант-ответ корректно отвечает на вопрос по тексту? "
                    "Ответь одним словом: 1 (если верно) или 0 (если неверно)."
                )
                enc = tok(prompt,
                          truncation=True,
                          max_length=max_len,
                          add_special_tokens=False)

                tgt_txt = LABEL_MUSERC.get(label, "")
                with tok.as_target_tokenizer():
                    tgt_ids = tok(tgt_txt,
                                  max_length=20,
                                  truncation=True).input_ids

                rows.append({
                    "input_ids":      enc.input_ids,
                    "attention_mask": enc.attention_mask,
                    "labels":         tgt_ids,
                    "qa_idx":         f"{pid}_{q_idx}_{a_idx}",
                    "idx":            pid
                })

    return Dataset.from_list(rows)



def build_rucos_dataset_t5(path: pathlib.Path, tok, max_len=512):
    rows = []
    for ex in Dataset.from_json(str(path)):
        passage_txt = ex["passage"]["text"]
        passage_ids = tok(passage_txt, add_special_tokens=False).input_ids

        for qa in ex["qas"]:
            q_idx = qa["idx"]

            answers = [
                {
                    "text":  a["text"],
                    "idx":   a.get("idx", i),
                    "label": a.get("label", 0)
                }
                for i, a in enumerate(qa.get("answers", []))
            ]

            if not answers:
                answers = [
                    {"text": passage_txt[e["start"]:e["end"]],
                    "idx":  i,
                    "label": 0}
                    for i, e in enumerate(ex["passage"]["entities"])
                ]

            if not answers:
                continue

            pos = next((a for a in answers if a["label"] == 1), answers[0])
            aid = pos["idx"]

            # --- формируем prompt и target -------------------------------
            prompt = (
                f"{passage_txt}\n\n{qa['query']}\n"
                "Подставь корректную сущность вместо @placeholder."
            )
            enc = tok(prompt, truncation=True, max_length=max_len)

            with tok.as_target_tokenizer():
                tgt_ids = tok(pos["text"], max_length=20, truncation=True).input_ids

            rows.append(
                {
                    "input_ids":      enc.input_ids,
                    "attention_mask": enc.attention_mask,
                    "labels":         tgt_ids,
                    "qa_idx": f"{q_idx}_{aid}",
                    "q_idx":  q_idx 
                }
            )
    return Dataset.from_list(rows)




def build_rucos_accuracy(tokenizer):
    pad_id = tokenizer.pad_token_id

    def _metric(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels == -100, pad_id, labels)

        p_txt = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tokenizer.batch_decode(labels, skip_special_tokens=True)

        p = np.array([TXT_RUCOS.get(t.strip().lower(), -1) for t in p_txt])
        y = np.array([TXT_RUCOS.get(t.strip().lower(), -1) for t in y_txt])

        return {"accuracy": float((p == y).mean())}

    return _metric





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
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=1)

    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)

    return accuracy.compute(
        predictions=preds.tolist(),
        references=labels.tolist()
    )



def compute_accuracy(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=1)
    return accuracy.compute(
        predictions=preds.tolist(),
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

def prepreprocess_parus(row: dict) -> dict | None:
    premise  = process_sentence(row.get("premise", ""))
    choice1  = process_sentence(row.get("choice1", ""))
    choice2  = process_sentence(row.get("choice2", ""))
    question = row.get("question", "").lower()

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
    if "label" in row:
        out["label"] = row["label"]
    return out



def preprocess_parus_t5(row, tok, max_len: int = 256):
    question = row["question"].strip().lower()
    premise   = row["premise"]
    c1        = row["choice1"]
    c2        = row["choice2"]

    if question in ("effect", "эффект") or question in ("cause", "причина"):
        src = (
            "Какая наиболее вероятная причина описанного в тексте события?\n"
            f"{premise}\n"
            f"0) {c1}\n"
            f"1) {c2}\n"
            "Ответ цифрой 0 или 1"
        )

    else:
        raise ValueError(f"Неизвестный тип question: {row['question']}")

    model_inputs = tok(src, truncation=True, max_length=max_len)

    label_id = row.get("label", -1)
    tgt_text = str(label_id) if label_id in (0, 1) else ""
    with tok.as_target_tokenizer():
        labels = tok(tgt_text, max_length=2, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def safe_int(txt: str) -> int:
    txt = txt.strip().lower()
    return 1 if txt in ("1", "effect", "эффект") else 0



def preprocess_terra_t5(row, tok, max_len=256):
    src = f"Следует из исходного высказывания ли данное утверждение: {process_sentence(row['premise'])}/n{process_sentence(row['hypothesis'])}. Ответь одним словом **entailment** (если следует) или **not_entailment** (если не следует)."
    model_inputs = tok(src, truncation=True, max_length=max_len)

    tgt = row.get("label", None)
    tgt_text = row["label"] if tgt is not None else ""
    with tok.as_target_tokenizer():
        labels = tok(tgt_text, max_length=20, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_lidirus_t5(row, tok, max_len=256):
    src = f"Как соотносятся два предложения: {process_sentence(row['sentence1'])}/n{process_sentence(row['sentence2'])}.  Ответь одним словом **entailment** (если следует друг другу) или **not_entailment** (противоречат друг другу)."
    model_inputs = tok(src, truncation=True, max_length=max_len)

    tgt_text = row.get("label", "")
    with tok.as_target_tokenizer():
        labels = tok(tgt_text, max_length=20, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_dnqa_t5(row, tok, max_len: int = 256):
    src = (
        f"Ответь да/нет на вопрос: {process_sentence(row['question'])} согласно тексту {process_sentence(row['passage'])}"
    )
    model_inputs = tok(src, truncation=True, max_length=max_len)

    lab = row.get("label", "")
    if isinstance(lab, bool):
        tgt_text = "true" if lab else "false"
    else:
        tgt_text = str(lab)

    with tok.as_target_tokenizer():
        labels = tok(tgt_text, max_length=20, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




def preprocess_rwsd_t5(row, tok, max_len=256):
    src = (
        f"В тексте {process_sentence(row['text'])}\n"
        f"Относилось ли фраза {process_sentence(row['target']['span2_text'])}\n"
        f"к объекту {process_sentence(row['target']['span1_text'])}\n"
        f"Ответь True (если относится) / False (если не относится)"
    )
    model_inputs = tok(src, truncation=True, max_length=max_len)

    if "label" in row:
        tgt_txt = "true" if str(row["label"]).lower() in ("true", "1") else "false"
    else:
        tgt_txt = ""
    with tok.as_target_tokenizer():
        labels = tok(tgt_txt, max_length=56, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



LABEL_RUSSE = {0: "false", 1: "true"}

def preprocess_russe_t5(ex, tok, max_len: int = 256):
    src = (
        "Выдели общее слово из двух предложений:\n"
        f"{process_sentence(ex['sentence1'])}\n"
        f"{process_sentence(ex['sentence2'])}"
    )

    model_inputs = tok(src, truncation=True, max_length=max_len)

    tgt_txt = LABEL_RUSSE.get(ex.get("label", -1), "")
    with tok.as_target_tokenizer():
        labels = tok(tgt_txt, max_length=20, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["idx"]    = ex["idx"]
    return model_inputs
