import re
from typing import Any
import numpy as np
import os
os.environ["HF_HOME"] = "/userspace/tev/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/userspace/tev/cache/"
os.environ["MPLCONFIGDIR"] = "/userspace/tev/cache/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_t5_accuracy(tokenizer):
    def _metric(eval_pred):
        preds, labels = eval_pred

        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        pred_txt  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        label_txt = tokenizer.batch_decode(labels,  skip_special_tokens=True)

        p = [LABEL_MAP_RCB.get(t.strip().lower(), -1) for t in pred_txt]
        y = [LABEL_MAP_RCB.get(t.strip().lower(), -1) for t in label_txt]
        acc = (np.array(p) == np.array(y)).mean()
        return {"accuracy": acc}
    return _metric

def quick_clean_rcb(ans: str) -> str:
    ans = ans.lower().strip()
    for lbl in ("contradiction", "entailment", "neutral"):
        if ans.startswith(lbl):
            return lbl
    return "neutral"          # fallback


def build_t5_accuracy_fast(tokenizer):
    pad = tokenizer.pad_token_id
    def _metric(ev_pred):
        preds, labels = ev_pred
        preds  = _sanitize_for_decode(preds,  pad)
        labels = _sanitize_for_decode(labels, pad)

        p_txt = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tokenizer.batch_decode(labels, skip_special_tokens=True)

        p = np.array([LABEL_MAP_RCB[quick_clean_rcb(t)] for t in p_txt])
        y = np.array([LABEL_MAP_RCB[quick_clean_rcb(t)] for t in y_txt])
        return {"accuracy": (p == y).mean()}
    return _metric



def quick_clean_parus(raw: str) -> int:
    t = raw.lower().strip()
    if t in ("0", "cause", "причина"):
        return 0
    if t in ("1", "effect", "эффект"):
        return 1
    if t and t[0] in "01":                    # вариант 1, ответ: 0
        return int(t[0])
    return -1


def build_parus_accuracy_fast(tok):
    def _metric(ev):
        preds, labels = ev
        preds  = _sanitize_for_decode(preds,  tok.pad_token_id)
        labels = _sanitize_for_decode(labels, tok.pad_token_id)

        p_txt = tok.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tok.batch_decode(labels, skip_special_tokens=True)

        p = np.array([quick_clean_parus(t) for t in p_txt])
        y = np.array([quick_clean_parus(t) for t in y_txt])

        return {"accuracy": float((p == y).mean())}
    return _metric



def quick_clean_tl(raw: str) -> str:
    t = raw.lower().strip()
    if t.startswith("entail"):
        return "entailment"
    if t.startswith("not") or t.startswith("contrad") or t.startswith("false"):
        return "not_entailment"
    if t and t[0] in "01":                 # «0», «1», «вариант 0»
        return "entailment" if t[0] == "0" else "not_entailment"
    return "not_entailment"                # fallback



def build_bin_accuracy_fast(tok):
    def _metric(ev):
        preds, labels = ev
        preds  = _sanitize_for_decode(preds,  tok.pad_token_id)
        labels = _sanitize_for_decode(labels, tok.pad_token_id)

        p_txt = tok.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tok.batch_decode(labels, skip_special_tokens=True)

        # приводим к 'entailment' / 'not_entailment'
        p = np.array([quick_clean_tl(t) for t in p_txt])
        y = np.array([quick_clean_tl(t) for t in y_txt])

        return {"accuracy": float((p == y).mean())}
    return _metric


def quick_clean_dnqa(raw: str) -> str:
    t = raw.lower().strip()
    # прямые ответы
    if t.startswith("true") or t in {"да", "верно", "правда", "yes"}:
        return "true"
    if t.startswith("false") or t in {"нет", "неверно", "ложь", "no"}:
        return "false"

    # одиночная 0 / 1
    if t and t[0] in "01":
        return "true" if t[0] == "1" else "false"

    return "false"


def build_dnqa_accuracy_fast(tok):
    pad = tok.pad_token_id

    def _metric(ev_pred):
        preds, labels = ev_pred
        preds  = _sanitize_for_decode(preds,  pad)
        labels = _sanitize_for_decode(labels, pad)

        p_txt = tok.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tok.batch_decode(labels, skip_special_tokens=True)

        p = np.array([quick_clean_dnqa(t) for t in p_txt])
        y = np.array([quick_clean_dnqa(t) for t in y_txt])

        return {"accuracy": float((p == y).mean())}

    return _metric



def quick_clean_rwsd(raw: str) -> str:
    t = raw.lower().strip()

    if t.startswith("true") or t in {"да", "верно", "истина", "yes"}:
        return "true"
    if t.startswith("false") or t in {"нет", "неверно", "ложь", "no"}:
        return "false"

    if t and t[0] in "01":              # «0», «1», «вариант 1»
        return "true" if t[0] == "1" else "false"

    return "false"                      # fallback


def build_rwsd_accuracy_fast(tok):
    pad = tok.pad_token_id

    def _metric(ev):
        preds, labels = ev
        preds  = _sanitize_for_decode(preds,  pad)
        labels = _sanitize_for_decode(labels, pad)

        p_txt = tok.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tok.batch_decode(labels, skip_special_tokens=True)

        p = np.array([quick_clean_rwsd(t) for t in p_txt])
        y = np.array([quick_clean_rwsd(t) for t in y_txt])

        return {"accuracy": float((p == y).mean())}

    return _metric



def quick_clean_muserc(raw: str) -> int:
    if isinstance(raw, int):
        return 1 if raw else 0

    t = raw.lower().strip()
    if t.startswith(("1", "true", "yes", "да", "верно")):
        return 1
    if t.startswith(("0", "false", "no", "нет", "неверно")):
        return 0
    return 0


def build_muserc_accuracy_fast(tok):
    pad = tok.pad_token_id

    def _metric(ev):
        preds, labels = ev
        preds  = _sanitize_for_decode(preds,  pad)
        labels = _sanitize_for_decode(labels, pad)

        p_txt = tok.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tok.batch_decode(labels, skip_special_tokens=True)

        p = np.array([quick_clean_muserc(t) for t in p_txt])
        y = np.array([quick_clean_muserc(t) for t in y_txt])

        return {"accuracy": float((p == y).mean())}

    return _metric



def quick_clean_rucos(raw: str) -> str:
    t = raw.lower().strip()

    if t.startswith(("true", "yes", "да")):
        return "true"
    if t.startswith(("false", "no", "нет")):
        return "false"
    if t and t[0] in "01":
        return "true" if t[0] == "1" else "false"
    return "false"



def quick_clean_russe(raw: str) -> str:
    t = raw.lower().strip()

    if t.startswith(("true", "да", "верно", "yes")):
        return "true"
    if t.startswith(("false", "нет", "неверно", "no")):
        return "false"
    if t and t[0] in "01":
        return "true" if t[0] == "1" else "false"

    return "false"


def build_rucos_accuracy_fast(tok):
    pad = tok.pad_token_id

    def _metric(ev):
        preds, labels = ev
        preds  = _sanitize_for_decode(preds,  pad)
        labels = _sanitize_for_decode(labels, pad)

        p_txt = tok.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tok.batch_decode(labels, skip_special_tokens=True)

        p = np.array([quick_clean_rucos(t) for t in p_txt])
        y = np.array([quick_clean_rucos(t) for t in y_txt])

        return {"accuracy": float((p == y).mean())}

    return _metric


def build_russe_accuracy_fast(tok):
    pad = tok.pad_token_id

    def _metric(ev):
        preds, labels = ev
        preds  = _sanitize_for_decode(preds,  pad)
        labels = _sanitize_for_decode(labels, pad)

        p_txt = tok.batch_decode(preds,  skip_special_tokens=True)
        y_txt = tok.batch_decode(labels, skip_special_tokens=True)

        p = np.array([quick_clean_russe(t) for t in p_txt])
        y = np.array([quick_clean_russe(t) for t in y_txt])

        return {"accuracy": float((p == y).mean())}

    return _metric

