import json
from datasets import Dataset
from matplotlib import pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["HF_HOME"] = "/userspace/tev/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/userspace/tev/cache/"
os.environ["MPLCONFIGDIR"] = "/userspace/tev/cache/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DISABLE_P2P"]= "1"

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from pathlib import Path
from transformers import AutoTokenizer

from datasets import Value
import numpy as np
import logging
import torch
from collections import defaultdict


from toolt5_full_sav import (
    build_muserc_dataset_t5,
    build_rucos_dataset_t5,
    prepreprocess_parus,
    preprocess_dnqa_t5,
    preprocess_lidirus_t5,
    preprocess_parus_t5,
    preprocess_rcb_t5,
    preprocess_russe_t5,
    preprocess_rwsd_t5,
    preprocess_terra_t5
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _sanitize_for_decode(arr, pad_id: int):
    import numpy as np

    if isinstance(arr, tuple):
        arr = arr[0]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.argmax(-1)
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64, copy=False)
    np.putmask(arr, arr < 0, pad_id)
    return arr



def dump_raw_test(task: str, texts: list[str], save_dir: str) -> None:
    path = Path(save_dir) / f"{task}_raw_test.log"
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(f"{i}\t{t}\n")



def to_tokens(preds):
    preds = preds[0] if isinstance(preds, tuple) else preds
    preds = preds.tolist() if isinstance(preds, np.ndarray) else preds
    if preds and isinstance(preds[0][0], (list, np.ndarray, tuple)):
        preds = [[int(np.argmax(tok)) for tok in seq] for seq in preds]
    return [[int(tok) for tok in seq] for seq in preds]


LABEL_MAP_RCB = {"contradiction": 0, "entailment": 1, "neutral": 2}
num_train_epochs = 20

from validate import (
    build_bin_accuracy_fast,
    build_dnqa_accuracy_fast,
    build_muserc_accuracy_fast,
    build_parus_accuracy_fast,
    build_rucos_accuracy_fast,
    build_russe_accuracy_fast,
    build_rwsd_accuracy_fast,
    build_t5_accuracy_fast,
    quick_clean_dnqa,
    quick_clean_muserc,
    quick_clean_rcb,
    quick_clean_rucos,
    quick_clean_russe,
    quick_clean_rwsd,
    quick_clean_tl,
    validate_dnqa_answer,
    validate_muserc_answer,
    validate_parus_answer,
    validate_rcb_answer,
    validate_rswd_answer,
    validate_rucos_answer,
    validate_terra_lidirus_answer,
    validate_russe_answer,
    quick_clean_parus,
)


def rcb(RSGLUE, tokenizer, SAVE_DIR, model_dir, device, SEED):
    logger.info("RCB")
    raw_train = Dataset.from_json(str(RSGLUE) + "RCB/train.jsonl")
    raw_val   = Dataset.from_json(str(RSGLUE) + "RCB/val.jsonl")
    raw_test  = Dataset.from_json(str(RSGLUE) + "RCB/test.jsonl")

    preprocess = lambda ex: preprocess_rcb_t5(ex, tokenizer)

    train_ds = (raw_train.map(preprocess, remove_columns=raw_train.column_names)
                         .filter(lambda x: x is not None))
    val_ds   = (raw_val  .map(preprocess, remove_columns=raw_val.column_names)
                         .filter(lambda x: x is not None))
    test_ds  = (raw_test .map(preprocess, remove_columns=raw_test.column_names)
                         .filter(lambda x: x is not None))

    start_id = tokenizer.pad_token_id
    test_ds = test_ds.map(lambda x: {"labels": [start_id]})

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(  type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda:0")
    args = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "rcb_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 12,
        per_device_eval_batch_size  = 12,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,

        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",

        predict_with_generate  = True,
        generation_max_length = 12,

        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )

    collator   = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric_fn = build_t5_accuracy_fast(tokenizer)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = metric_fn
    )

    trainer.train()

    val_out   = trainer.predict(val_ds, max_length=args.generation_max_length)

    pred_ids  = _sanitize_for_decode(val_out.predictions, start_id)
    label_ids = _sanitize_for_decode(val_out.label_ids,  start_id)

    pred_txt  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    gold_txt  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_txt_clean  = [quick_clean_rcb(t) for t in pred_txt]
    gold_txt_clean  = [quick_clean_rcb(t) for t in gold_txt]

    p_ids = np.array([LABEL_MAP_RCB.get(lbl, -1) for lbl in pred_txt_clean])
    g_ids = np.array([LABEL_MAP_RCB.get(lbl, -1) for lbl in gold_txt_clean])



    logger.info("\n=== first 3 validation samples ===")
    for i in range(3):
        p = pred_txt_clean[i]
        g = gold_txt_clean[i]
        logger.info(f"[{i}] PRED: {p:13} ({p_ids[i]}) | GOLD: {g:13} ({g_ids[i]})")
    logger.info("===================================\n")

    eval_acc = (p_ids == g_ids).mean()
    logger.info(f"RCB eval accuracy: {eval_acc:.4f}")

    pred_test_ids = _sanitize_for_decode(
        trainer.predict(test_ds).predictions,
        start_id
    )
    gen_texts = tokenizer.batch_decode(pred_test_ids, skip_special_tokens=True)
    
    dump_raw_test("RCB", gen_texts, SAVE_DIR)


    results = []
    for i, txt in enumerate(gen_texts):
        clean_lbl = quick_clean_rcb(
            txt
        )
        results.append({"idx": i, "label": clean_lbl})

    with open(SAVE_DIR + "RCB.jsonl", "w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info("RCB Done")


def parus(RSGLUE, SAVE_DIR, model_dir, device, SEED, tokenizer):
    logger.info("PARus")
    raw_train = Dataset.from_json(str(RSGLUE) + "PARus/train.jsonl").map(prepreprocess_parus)
    raw_val   = Dataset.from_json(str(RSGLUE) + "PARus/val.jsonl").map(  prepreprocess_parus)
    raw_test  = Dataset.from_json(str(RSGLUE) + "PARus/test.jsonl").map( prepreprocess_parus)

    proc = lambda x: preprocess_parus_t5(x, tokenizer)
    train_ds = raw_train.map(proc, remove_columns=raw_train.column_names)
    val_ds   = raw_val  .map(proc, remove_columns=raw_val.column_names)
    test_ds  = raw_test .map(proc, remove_columns=raw_test.column_names)
    start_id = tokenizer.pad_token_id
    test_ds = test_ds.map(lambda x: {"labels": [start_id]})

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(  type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    logger.info(f"PARus processed -> train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir
    ).to("cuda:0")

    args = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "parus_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 12,
        per_device_eval_batch_size  = 12,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,

        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",

        predict_with_generate  = True,
        generation_max_length = 12,      

        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )

    collator   = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric_fn = build_parus_accuracy_fast(tokenizer)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = metric_fn,
    )

    trainer.train()

    val_out   = trainer.predict(val_ds, max_length=args.generation_max_length)

    pred_ids  = _sanitize_for_decode(val_out.predictions, start_id)
    label_ids = _sanitize_for_decode(val_out.label_ids,  start_id)

    raw_pred_txt = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    gold_txt     = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    clean_pred = [quick_clean_parus(t) for t in raw_pred_txt]
    gold_int   = [quick_clean_parus(t) for t in gold_txt]


    logger.info("\n=== first 3 validation samples (PARus) ===")
    for i in range(3):
        logger.info(f"[{i}] PRED(raw) {raw_pred_txt[i]!r}  →  {clean_pred[i]}"
                    f" | GOLD {gold_int[i]}")
    logger.info("===========================================\n")

    eval_acc = (np.array(clean_pred) == np.array(gold_int)).mean()
    logger.info(f"PARus val-accuracy: {eval_acc:.4f}")

    test_ids = _sanitize_for_decode(
        trainer.predict(test_ds).predictions,
        start_id
    )

    test_raw   = tokenizer.batch_decode(test_ids, skip_special_tokens=True)
    dump_raw_test("PARus", test_raw, SAVE_DIR)
    
    test_clean = [quick_clean_parus(t) for t in test_raw]

    results = [{"idx": i, "label": int(lbl)} for i, lbl in enumerate(test_clean)]

    out_path = Path(SAVE_DIR) / "PARus.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in results:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    logger.info("PARus Done")


def terra_lidirus(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer):
    logger.info("TERRa")

    raw_train = Dataset.from_json(str(RSGLUE) + "TERRa/train.jsonl")
    raw_val   = Dataset.from_json(str(RSGLUE) + "TERRa/val.jsonl")
    raw_test  = Dataset.from_json(str(RSGLUE) + "TERRa/test.jsonl")

    tv_proc = lambda x: preprocess_terra_t5(x, tokenizer)
    train_ds = raw_train.map(tv_proc, remove_columns=raw_train.column_names)
    val_ds   = raw_val  .map(tv_proc, remove_columns=raw_val.column_names)
    test_ds  = raw_test .map(tv_proc, remove_columns=raw_test.column_names)

    pad_id = tokenizer.pad_token_id
    test_ds = test_ds.map(lambda _: {"labels": [pad_id]})

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(  type="torch", columns=cols)
    test_ds.set_format( type="torch", columns=cols)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda:0")

    args = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "terra_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 10,
        per_device_eval_batch_size  = 10,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,

        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",

        predict_with_generate  = True,
        generation_max_length  = 12,

        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )

    collator   = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric_fn = build_bin_accuracy_fast(tokenizer)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = metric_fn,
        tokenizer       = tokenizer,
    )

    trainer.train()
    torch.cuda.empty_cache()

    val_pred = trainer.predict(val_ds, max_length=args.generation_max_length)

    pred_ids  = _sanitize_for_decode(val_pred.predictions, pad_id)
    label_ids = _sanitize_for_decode(val_pred.label_ids,  pad_id)
    raw_pred_txt = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    gold_txt     = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    clean_pred  = [quick_clean_tl(t) for t in raw_pred_txt]
    clean_gold  = [quick_clean_tl(t) for t in gold_txt]


    to_int = lambda t: 0 if t == "entailment" else 1

    logger.info("\n=== TERRa | первые 3 примера ===")
    for i in range(3):
        p, g = clean_pred[i], clean_gold[i]
        logger.info(f"[{i}] PRED: {p:15} -> {to_int(p)} | GOLD: {g:15} -> {to_int(g)}")
    logger.info("================================\n")

    eval_acc = (np.array([to_int(p) for p in clean_pred]) ==
                np.array([to_int(g) for g in clean_gold])).mean()
    logger.info(f"filtered TERRA val-accuracy: {eval_acc:.4f}")

    test_ids = _sanitize_for_decode(
        trainer.predict(test_ds).predictions,
        pad_id 
    )

    test_raw = tokenizer.batch_decode(test_ids, skip_special_tokens=True)
    dump_raw_test("TERRa", test_raw, SAVE_DIR)

    terra_out = [
        {"idx": i, "label": quick_clean_tl(t)}
        for i, t in enumerate(test_raw)
    ]



    with open(SAVE_DIR + "TERRa.jsonl", "w", encoding="utf-8") as f:
        for obj in terra_out:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
    logger.info("TERRa Done")



    logger.info("LiDiRus")
    lid_raw = Dataset.from_json(str(RSGLUE) + "LiDiRus/LiDiRus.jsonl")
    lid_ds  = lid_raw.map(lambda x: preprocess_lidirus_t5(x, tokenizer),
                        remove_columns=lid_raw.column_names
            ).map(lambda _: {"labels": [pad_id]}) \
            .with_format("torch", columns=["input_ids","attention_mask","labels"])

    _saved = trainer.compute_metrics
    trainer.compute_metrics = None

    lid_pred_tok = trainer.predict(lid_ds).predictions
    lid_pred_txt = tokenizer.batch_decode(to_tokens(lid_pred_tok),
                                        skip_special_tokens=True)

    sample_tok   = trainer.predict(lid_ds.select(range(3))).predictions
    sample_txt   = tokenizer.batch_decode(to_tokens(sample_tok),
                                        skip_special_tokens=True)
    
    trainer.compute_metrics = _saved

    lid_out = [
        {"idx": int(lid_raw[i]["idx"]),
        "label": quick_clean_tl(t)}
        for i, t in enumerate(lid_pred_txt)
    ]

    logger.info("=== LiDiRus | первые 3 примера ===")
    for i, txt in enumerate(sample_txt):
        logger.info(f"[{i}] PRED: {txt.lower().strip()}")
    logger.info("==================================")

    with open(SAVE_DIR / "LiDiRus.jsonl", "w", encoding="utf-8") as f:
        for obj in lid_out:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    logger.info("LiDiRus Done")



def DaNetQA(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer):
    logger.info("DaNetQA")
    raw_train = Dataset.from_json(str(RSGLUE) + "DaNetQA/train.jsonl")
    raw_val   = Dataset.from_json(str(RSGLUE) + "DaNetQA/val.jsonl")
    raw_test  = Dataset.from_json(str(RSGLUE) + "DaNetQA/test.jsonl")

    proc = lambda x: preprocess_dnqa_t5(x, tokenizer)
    train_ds = raw_train.map(proc, remove_columns=raw_train.column_names)
    val_ds   = raw_val  .map(proc, remove_columns=raw_val.column_names)
    test_ds  = raw_test .map(proc, remove_columns=raw_test.column_names)

    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds   = val_ds.filter(  lambda x: x is not None)
    test_ds  = test_ds.filter( lambda x: x is not None)
    
    start_id = tokenizer.pad_token_id
    test_ds = test_ds.map(lambda x: {"labels": [start_id]})

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(  type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda:0")

    args = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "danetqa_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 12,
        per_device_eval_batch_size  = 12,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,

        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",

        predict_with_generate  = True,
        generation_max_length = 12,      

        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )

    collator   = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric_fn = build_dnqa_accuracy_fast(tokenizer)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = metric_fn,
    )

    trainer.train()
    torch.cuda.empty_cache()

    val_pred = trainer.predict(val_ds, max_length=args.generation_max_length)

    pred_ids  = _sanitize_for_decode(val_pred.predictions, start_id)
    label_ids = _sanitize_for_decode(val_pred.label_ids,  start_id)

    raw_pred_txt = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    gold_txt     = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    clean_pred = [quick_clean_dnqa(t) for t in raw_pred_txt]
    clean_gold = [quick_clean_dnqa(t) for t in gold_txt]


    to_int = lambda t: 1 if t == "true" else 0

    logger.info("\n=== DaNetQA | первые 3 вал-примера ===")
    for i in range(3):
        p, g = clean_pred[i], clean_gold[i]
        logger.info(f"[{i}] PRED {p:5}→{to_int(p)} | GOLD {g:5}→{to_int(g)}")
    logger.info("======================================\n")

    eval_acc = (np.array([to_int(p) for p in clean_pred]) ==
                np.array([to_int(g) for g in clean_gold])).mean()
    logger.info(f"DaNetQA val-accuracy: {eval_acc:.4f}")

    test_ids = _sanitize_for_decode(
        trainer.predict(test_ds).predictions,
        start_id
    )

    test_raw = tokenizer.batch_decode(test_ids, skip_special_tokens=True)
    dump_raw_test("DaNetQA", test_raw, SAVE_DIR)

    out = [
        {"idx": i, "label": build_dnqa_accuracy_fast(txt)}
        for i, txt in enumerate(test_raw)
    ]

    with open(SAVE_DIR + "DaNetQA.jsonl", "w", encoding="utf-8") as f:
        for obj in out:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    logger.info("DaNetQA Done")



def RWSD(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer):
    logger.info("RWSD")

    raw_train = Dataset.from_json(str(RSGLUE) + "RWSD/train.jsonl")
    raw_val   = Dataset.from_json(str(RSGLUE) + "RWSD/val.jsonl")
    raw_test  = Dataset.from_json(str(RSGLUE) + "RWSD/test.jsonl")

    proc = lambda x: preprocess_rwsd_t5(x, tokenizer)
    train_ds = raw_train.map(proc, remove_columns=raw_train.column_names)
    val_ds   = raw_val  .map(proc, remove_columns=raw_val.column_names)
    test_ds  = raw_test .map(proc, remove_columns=raw_test.column_names)

    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds   = val_ds.filter(  lambda x: x is not None)
    test_ds  = test_ds.filter( lambda x: x is not None)
    
    start_id = tokenizer.pad_token_id
    test_ds = test_ds.map(lambda x: {"labels": [start_id]})

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(  type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda:0")

    args = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "rwsd_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 12,
        per_device_eval_batch_size  = 12,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,

        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",

        predict_with_generate  = True,
        generation_max_length = 12,      

        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )

    collator  = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric_fn = build_rwsd_accuracy_fast(tokenizer)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = metric_fn,
    )

    trainer.train()
    torch.cuda.empty_cache()

    val_pred = trainer.predict(val_ds, max_length=args.generation_max_length)


    pred_ids  = _sanitize_for_decode(val_pred.predictions, start_id)
    label_ids = _sanitize_for_decode(val_pred.label_ids,  start_id)

    raw_pred_txt = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    gold_txt     = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    clean_pred = [quick_clean_rwsd(t) for t in raw_pred_txt]
    clean_gold = [quick_clean_rwsd(t) for t in gold_txt]

    to_int = lambda t: 1 if t == "True" else 0

    logger.info("\n=== RSWD | первые 3 вал-примера ===")
    for i in range(3):
        p, g = clean_pred[i], clean_gold[i]
        logger.info(f"[{i}] PRED {p:5}→{to_int(p)} | GOLD {g:5}→{to_int(g)}")
    logger.info("======================================\n")

    eval_acc = (np.array([to_int(p) for p in clean_pred]) ==
                np.array([to_int(g) for g in clean_gold])).mean()
    logger.info(f"filtered RSWD val-accuracy: {eval_acc:.4f}")

    raw_pred = trainer.predict(test_ds).predictions
    test_ids = _sanitize_for_decode(raw_pred, start_id)
    test_txt = tokenizer.batch_decode(test_ids, skip_special_tokens=True)
    
    dump_raw_test("RWSD", test_txt, SAVE_DIR)

    out = [
        {"idx": i, "label": quick_clean_rwsd(txt)}
        for i, txt in enumerate(test_txt)
    ]

    with open(SAVE_DIR + "RSWD.jsonl", "w", encoding="utf-8") as f:
        for obj in out:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    logger.info("RSWD Done")



def MuSeRC(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer):
    logger.info("MuSeRC")

    build   = lambda p: build_muserc_dataset_t5(p, tokenizer)
    train_ds = build(Path(RSGLUE) / "MuSeRC/train.jsonl")
    val_ds   = build(Path(RSGLUE) / "MuSeRC/val.jsonl")
    test_ds  = build(Path(RSGLUE) / "MuSeRC/test.jsonl")
    
    pad_id   = tokenizer.pad_token_id          
    test_ds  = test_ds.map(lambda _: {"labels": [pad_id]})

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols + ["qa_idx"])
    val_ds.set_format(  type="torch", columns=cols + ["qa_idx"])

    test_ds.set_format(type="torch",
                    columns=["input_ids", "attention_mask",
                                "labels", "qa_idx"])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda:0")

    args = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "muserc_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 8,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,

        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",

        predict_with_generate  = True,
        generation_max_length = 12,      

        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )

    collator  = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric_fn = build_muserc_accuracy_fast(tokenizer)


    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = metric_fn,
    )

    trainer.train()
    torch.cuda.empty_cache()

    val_out = trainer.predict(val_ds, max_length=args.generation_max_length)

    pred_ids  = _sanitize_for_decode(val_out.predictions, pad_id)
    label_ids = _sanitize_for_decode(val_out.label_ids,  pad_id)

    pred_txt  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    gold_txt  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_int  = [quick_clean_muserc(t) for t in pred_txt]
    gold_int  = [quick_clean_muserc(t) for t in gold_txt]



    def to_int_bool(t) -> int:
        if type(t) == int:
            return int(t)
        else:
            return 1 if t.strip().lower() in ("1", "true", "yes") else 0

    logger.info("\n=== MuSeRC | первые 3 candidate-примера ===")
    for i in range(3):
        qid, aid = val_ds[i]["qa_idx"].split("_")
        p_raw, g_raw = pred_int[i], gold_int[i]
        p, g = to_int_bool(p_raw), to_int_bool(g_raw)
        logger.info(f"[{i}] Q{qid}-A{aid}: PRED {p} (raw '{p_raw}') | "
            f"GOLD {g} (raw '{g_raw}')")
    logger.info("===========================================\n")

    eval_acc = (np.array(pred_int) == np.array(gold_int)).mean()
    logger.info(f"MuSeRC eval accuracy (after validation): {eval_acc:.4f}")

    raw_pred = trainer.predict(test_ds).predictions
    test_ids = _sanitize_for_decode(raw_pred, pad_id)
    gen_txt  = tokenizer.batch_decode(test_ids, skip_special_tokens=True)
    
    dump_raw_test("MuSeRC", gen_txt, SAVE_DIR)

    gen_int = [quick_clean_muserc(t) for t in gen_txt]

    out = []
    grouped = defaultdict(lambda: defaultdict(list))

    for i, lbl in enumerate(gen_int):
        parts = test_ds[i]["qa_idx"].split("_")
        if len(parts) == 3:
            pid, qid, aid = map(int, parts)
        else:
            qid, aid = map(int, parts)
            pid = int(test_ds[i]["idx"])

        grouped[pid][qid].append({"idx": aid, "label": lbl})

    with open(Path(SAVE_DIR) / "MuSeRC.jsonl", "w", encoding="utf-8") as fout:
        for pid in sorted(grouped):
            out_questions = []
            for qid in sorted(grouped[pid]):
                answers = sorted(grouped[pid][qid], key=lambda x: x["idx"])
                out_questions.append({"idx": qid, "answers": answers})

            json.dump({"idx": pid, "passage": {"questions": out_questions}},
                    fout, ensure_ascii=False)
            fout.write("\n")

    logger.info("MuSeRC Done")



def RuCoS(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer):
    logger.info("RuCoS")

    build = lambda p: build_rucos_dataset_t5(p, tokenizer)
    train_ds = build(Path(RSGLUE) / "RuCoS/train.jsonl")
    val_ds   = build(Path(RSGLUE) / "RuCoS/val.jsonl")
    test_ds  = build(Path(RSGLUE) / "RuCoS/test.jsonl")
    logger.info(f"RuCoS processed: train={len(train_ds)}, "
                f"val={len(val_ds)}, test={len(test_ds)}")

    pad_id = tokenizer.pad_token_id
    test_ds = test_ds.map(lambda ex: {**ex, "labels": [pad_id]})

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols + ["qa_idx"])
    val_ds.set_format(  type="torch", columns=cols + ["qa_idx"])
    test_ds.set_format(type="torch",
                       columns=["input_ids", "attention_mask", "labels", "qa_idx", "q_idx"])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda:0")
    args  = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "rucos_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 8,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,
        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",
        predict_with_generate  = True,
        generation_max_length  = 12,
        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )
    compute_metric = build_rucos_accuracy_fast(tokenizer)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics = compute_metric
    )

    trainer.train()
    torch.cuda.empty_cache()

    val_pred = trainer.predict(val_ds, max_length=args.generation_max_length)

    pred_ids  = _sanitize_for_decode(val_pred.predictions, pad_id)
    label_ids = _sanitize_for_decode(val_pred.label_ids,  pad_id)

    pred_txt = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    gold_txt = tokenizer.batch_decode(label_ids, skip_special_tokens=True)


    logger.info("\n=== RuCoS | первые 3 candidate-ответа (val) ===")
    for i in range(3):
        qid, aid = val_ds[i]["qa_idx"].split("_")
        clean_p  = quick_clean_rucos(pred_txt[i])
        logger.info(f"[{i}] Q{qid}-A{aid}: PRED «{clean_p}» | RAW «{pred_txt[i]}» | "
                    f"GOLD {gold_txt[i]}")


    eval_acc = trainer.evaluate()["eval_accuracy"]
    logger.info(f"RuCoS eval accuracy: {eval_acc:.4f}")

    raw_pred = trainer.predict(test_ds).predictions
    test_ids = _sanitize_for_decode(raw_pred, pad_id)
    raw_txt  = tokenizer.batch_decode(test_ids, skip_special_tokens=True)
    
    dump_raw_test("RuCoS", raw_txt, SAVE_DIR)

    results = [
        {"idx": int(test_ds[i]["q_idx"]),
        "label": quick_clean_rucos(raw_txt[i])}
        for i in range(len(test_ds))
    ]

    out_path = Path(SAVE_DIR) / "RuCoS.jsonl"
    with open(out_path,"w",encoding="utf-8") as f:
        for obj in results:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    logger.info("RuCoS Done")



def RUSSE(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer):
    logger.info("RUSSE")

    root = Path(RSGLUE)
    raw_train = Dataset.from_json(str(root / "RUSSE" / "train.jsonl"))
    raw_val   = Dataset.from_json(str(root / "RUSSE" / "val.jsonl"))
    raw_test  = Dataset.from_json(str(root / "RUSSE" / "test.jsonl"))

    proc = lambda x: preprocess_russe_t5(x, tokenizer)
    train_ds = raw_train.map(proc, remove_columns=raw_train.column_names)
    val_ds   = raw_val  .map(proc, remove_columns=raw_val.column_names)
    test_ds  = raw_test .map(proc, remove_columns=raw_test.column_names)

    pad_id = tokenizer.pad_token_id
    test_ds = test_ds.map(lambda _: {"labels": [pad_id]})

    cols = ["input_ids", "attention_mask", "labels", "idx"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(  type="torch", columns=cols)
    test_ds.set_format( type="torch", columns=cols)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda:0")

    args = Seq2SeqTrainingArguments(
        output_dir             = SAVE_DIR + "russe_cp",
        num_train_epochs       = num_train_epochs,
        per_device_train_batch_size = 12,
        per_device_eval_batch_size  = 12,
        learning_rate          = 1e-5,
        weight_decay           = 0.01,
        warmup_ratio           = 0.02,
        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "accuracy",
        predict_with_generate  = True,
        generation_max_length  = 12,
        fp16                   = (device.type != "cpu"),
        seed                   = SEED,
        report_to              = "none",
        dataloader_pin_memory = False,
    )
    
    compute_metric = build_russe_accuracy_fast(tokenizer)


    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics = compute_metric
    )

    trainer.train()
    torch.cuda.empty_cache()

    val_pred_tok = trainer.predict(val_ds).predictions
    val_pred_tok = _sanitize_for_decode(val_pred_tok, pad_id)

    val_pred_txt   = tokenizer.batch_decode(val_pred_tok, skip_special_tokens=True)
    val_pred_clean = [quick_clean_russe(t) for t in val_pred_txt]

    
    gold_clean = [
        quick_clean_russe(
            tokenizer.decode(
                [tok if tok != -100 else pad_id for tok in seq],
                skip_special_tokens=True,
            )
        )
        for seq in val_ds["labels"]
    ]

    acc = np.mean([p == g for p, g in zip(val_pred_clean, gold_clean)])
    logger.info(f"RUSSE offline clean-val accuracy: {acc:.4f}")


    raw_pred = trainer.predict(test_ds).predictions
    test_ids = _sanitize_for_decode(raw_pred, pad_id)
    test_txt = tokenizer.batch_decode(test_ids, skip_special_tokens=True)
    
    dump_raw_test("RUSSE", test_txt, SAVE_DIR)


    test_clean = [quick_clean_russe(t) for t in test_txt]

    
    out_path = Path(SAVE_DIR) / "RUSSE.jsonl"

    with open(out_path,"w",encoding="utf-8") as f:
        for idx,lbl in zip(test_ds["idx"], test_clean):
            json.dump({"idx": int(idx), "label": lbl}, f, ensure_ascii=False)
            f.write("\n")

    logger.info("RUSSE Done")


def main(
    tokenizer_dir: str,
    model_dir: str,
    rsglue_dir: str,
    output_dir: str
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    logger.info('tokenizer was loaded')
    logger.info('cls_data_collator was defined')
    SEED = 42
    RSGLUE = rsglue_dir
    SAVE_DIR = output_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rcb(RSGLUE, tokenizer, SAVE_DIR, model_dir, device, SEED)
    
    parus(RSGLUE, SAVE_DIR, model_dir, device, SEED, tokenizer)

    terra_lidirus(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer)
    
    DaNetQA(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer)
    
    RWSD(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer)
    
    MuSeRC(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer)
    
    RuCoS(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer)
    
    RUSSE(RSGLUE, SAVE_DIR, device, model_dir, SEED, tokenizer)



def to_token_ids_lidirus(preds) -> list[list[int]]:
    preds = preds[0] if isinstance(preds, tuple) else preds
    if preds.ndim == 3:
        preds = preds.argmax(-1)
    return [seq.tolist() if isinstance(seq, np.ndarray) else list(seq)
            for seq in preds]



if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))
    logger.addHandler(c_handler)
    logger.setLevel(logging.INFO)

    tokenizer = "/userspace/tev/cache/local-fredt5-model" # /userspace/tev/cache/checkpoints/fredt5-10k or /userspace/tev/cache/checkpoints/fredt5-1bln
    model = "/userspace/tev/cache/local-fredt5-model" # /userspace/tev/cache/checkpoints/fredt5-10k or /userspace/tev/cache/checkpoints/fredt5-1bln
    rsglue = './combined/'
    output = '/userspace/tev/cache/newstrat/20_mfredt5_10k' # other folders

    main(tokenizer_dir=tokenizer, model_dir=model, rsglue_dir=rsglue, output_dir=output)