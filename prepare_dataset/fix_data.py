"""
Стрим-обработка JSONL с батчевым inference на A100-80 GB.
Проверено: yandex/YandexGPT-5-Lite-8B-instruct, BF16, Flash-Attention; batch ≈ 64
"""


import os, json, re, logging, itertools, gc
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


LOCAL_MODEL = "/userspace/tev/cache/yandex"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.pad_token_id  = tokenizer.eos_token_id

torch.cuda.set_device(0)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)

from transformers import PreTrainedModel
assert isinstance(model, PreTrainedModel)


model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
torch.backends.cuda.matmul.allow_tf32 = True


_AS_TAG = re.compile(r"Ассистент:\[SEP\]", re.I)
_HDR    = re.compile(r"^[^\n:]{0,40}:\s*")

def _clean_assistant(txt: str) -> str:
    m = _AS_TAG.search(txt)
    if m:
        txt = txt[m.end():]
    return _HDR.sub("", txt, count=1).strip()

def _first_line(x: str) -> str:
    for ln in x.splitlines():
        ln = ln.strip().rstrip(".")
        if ln: 
            return ln
    return x.strip()


FEWSHOT = "Выведи только финальный ответ."

MC_SET = {
    'education', 'human interest', 'society', 'sport',
    'crime, law and justice', 'disaster, accident and emergency incident',
    'arts, culture, entertainment and media', 'politics',
    'economy, business and finance', 'lifestyle and leisure',
    'science and technology', 'health', 'labour', 'religion',
    'weather', 'environment', 'conflict, war and peace'
}


MC_PROMPT = (
    "Проверь, соответствует ли предложенная тематика содержанию текста.\n\n"
    "Выбери одну тематику из списка:\n{classes}\n\n"
    "ТЕКСТ СТАТЬИ:\n{article}\n\n"
    "Текущая тематика: {label}\n\n"
    "{fewshot}"
)

NLI_SET = {"entailment", "contradiction", "neutral"}

NLI_PROMPT = (
    "Определи отношение между premise и hypothesis. Возможные варианты: "
    "entailment, contradiction, neutral.\n\n"
    "PREMISE:\n{premise}\n\n"
    "HYPOTHESIS:\n{hypo}\n\n"
    "Текущая метка: {label}\n\n"
    "{fewshot}"
)


QA_QUESTION_PROMPT = (
    "Проверь, правильно ли сформулирован вопрос по тексту."
    "Если вопрос некорректен или ответа на него нет в тексте, переформулируй — задай другой вопрос, ответ на который действительно присутствует в тексте (не более 50 слов)."
    "Обязательно заканчивай итоговое предложение знаком вопроса «?».\n\n"
    "ТЕКСТ:\n{text}\n\n"
    "ВОПРОС:\n{question}\n\n"
    "{fewshot}"
)

QA_ANSWER_PROMPT = (
    "Ответь кратко на вопрос, опираясь только на информацию из текста (не более 50 слов).\n\n"
    "ТЕКСТ:\n{text}\n\n"
    "ВОПРОС:\n{question}\n\n"
    "{fewshot}"
)

MAX_NEW   = 48
GEN_BATCH = 64


def _match_from_set(cand: str, allowed: set[str]) -> str | None:
    cand_low = cand.lower()
    for item in allowed:
        if cand_low == item.lower():
            return item
    return None


def chat_many(prompts: list[str]) -> list[str]:
    tmpl = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True)
            for p in prompts]

    toks = tokenizer(
        tmpl, return_tensors="pt",
        padding=True, return_attention_mask=True, truncation=True
    ).to(model.device)

    with torch.no_grad():
        outs = model.generate(
            **toks,
            max_new_tokens=MAX_NEW,
            do_sample=False
        )
    return tokenizer.batch_decode(outs, skip_special_tokens=True)


def build_prompts(entry: Dict) -> list[tuple[str, str, Dict]]:
    tasks = []
    if "mc" in entry:
        tasks.append((MC_PROMPT.format(
            classes="\n".join(sorted(MC_SET)),
            article=entry['text'],
            label=entry['mc'],
            fewshot=FEWSHOT
        ), "mc", entry))
    if "nli" in entry:
        hypo = entry.get("pa") or entry.get("title", "")
        tasks.append((NLI_PROMPT.format(
            premise=entry['text'][:2000],
            hypo=hypo[:500],
            label=entry['nli'],
            fewshot=FEWSHOT
        ), "nli", entry))
    if "qg" in entry:
        tasks.append((QA_QUESTION_PROMPT.format(
            text=entry['text'][:2000],
            question=entry['qg'],
            fewshot=FEWSHOT
        ), "qg", entry))
        tasks.append((QA_ANSWER_PROMPT.format(
            text=entry['text'][:2000],
            question=entry['qg'],
            fewshot=FEWSHOT
        ), "qa", entry))
    return tasks

def clean_inplace(path: str, batch_size: int = 2048):
    total = sum(1 for _ in open(path, encoding="utf-8"))
    tmp_out = f"{path}.tmp"

    with open(path, encoding="utf-8") as fin, \
         open(tmp_out, "w", encoding="utf-8") as fout, \
         tqdm(total=total, desc="Fix", unit="doc", ncols=90) as bar:

        while True:
            raw_lines = list(itertools.islice(fin, batch_size))
            if not raw_lines:
                break

            entries = [json.loads(l) for l in raw_lines]

            jobs: list[tuple[str, str, Dict]] = []
            for e in entries:
                jobs.extend(build_prompts(e))

            answers: list[str] = []
            for i in range(0, len(jobs), GEN_BATCH):
                mini = jobs[i:i+GEN_BATCH]
                answers = chat_many([p for p, *_ in mini])
                for (_prompt, tag, entry), ans in zip(mini, answers):
                    ans = _first_line(_clean_assistant(ans))
                    if tag == "mc":
                        ok = _match_from_set(ans, MC_SET)
                        if ok: entry["mc"] = ok
                    elif tag == "nli":
                        ok = _match_from_set(ans, NLI_SET)
                        if ok: entry["nli"] = ok
                    elif tag == "qg":
                        entry["qg"] = ans
                    elif tag == "qa":
                        entry["qa"] = ans

            for e in entries:
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")

            bar.update(len(entries))
            del entries, jobs, raw_lines, answers
            gc.collect(); torch.cuda.empty_cache()

    os.replace(tmp_out, path)
    print(f"Файл перезаписан: {path}")



logging.basicConfig(
    filename="train.txt", filemode="a", encoding="utf-8", # val.txt
    level=logging.INFO, format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _apply_answer(tag: str, ans: str, entry: Dict):
    ans = _first_line(_clean_assistant(ans))
    if tag == "mc":
        ok = _match_from_set(ans, MC_SET)
        if ok:
            entry["mc"] = ok
            logger.info(f"[MC]\n{ok}\n")
    elif tag == "nli":
        ok = _match_from_set(ans, NLI_SET)
        if ok:
            entry["nli"] = ok
            logger.info(f"[NLI]\n{ok}\n")
    elif tag == "qg":
        entry["qg"] = ans
        logger.info(f"[QG]\n{ans}\n")
    elif tag == "qa":
        entry["qa"] = ans
        logger.info(f"[QA]\n{ans}\n")




def clean_jsonl(
    src: str,
    dst: str,
    read_batch: int = 2048,
    gen_batch: int = GEN_BATCH,
    fsync: bool = False
) -> None:
    src_p, dst_p = Path(src), Path(dst)
    total = sum(1 for _ in open(src_p, encoding="utf-8"))

    with open(src_p, encoding="utf-8") as fin, \
         open(dst_p, "a", encoding="utf-8", buffering=1) as fout, \
         tqdm(total=total, desc="Fix", ncols=90, unit="doc") as bar:

        while True:
            raw_lines = list(itertools.islice(fin, read_batch))
            if not raw_lines:
                break

            entries = [json.loads(l) for l in raw_lines]

            tasks: List[Tuple[str, str, Dict]] = []
            for e in entries:
                tasks.extend(build_prompts(e))

            for i in range(0, len(tasks), gen_batch):
                mini   = tasks[i : i + gen_batch]
                prompts = [p for p, *_ in mini]
                answers = chat_many(prompts)
                for (_p, tag, entry), ans in zip(mini, answers):
                    _apply_answer(tag, ans, entry)

            for e in entries:
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")

            fout.flush()
            if fsync:
                os.fsync(fout.fileno())

            bar.update(len(entries))
            del entries, tasks, raw_lines, answers
            gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    clean_jsonl(
        src="/userspace/tev/cache/checkpoints/data_1bln/train.jsonl", # .../val.jsonl
        dst="/userspace/tev/cache/checkpoints/data_1bln/train_fixed.jsonl", # .../val_out.jsonl
        read_batch=2048,
        gen_batch=GEN_BATCH,
    )
