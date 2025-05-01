import json
from transformers import AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer, DataCollatorWithPadding
from datasets import Dataset

import numpy as np
import logging
import argparse
import torch

import os

from tools_t5 import (
    DataCollatorForMultipleChoice,
    compute_mc_accuracy,
    compute_accuracy,
    seed_everything,
    preprocess_rcb,
    prepreprocess_parus,
    preprocess_terra,
    preprocess_lidirus,
    preprocess_dnqa,
    preprocess_rwsd,
    preprocess_parus_binary
)

# os.environ["HF_HOME"] = "/userspace/tev/cache/"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/userspace/tev/cache/"
# os.environ["MPLCONFIGDIR"] = "/userspace/tev/cache/"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_accuracy_parus(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"eval_accuracy": (preds == labels).mean()}



# ─────────────────────── main ──────────────────────────────────────
LABEL_MAP_RCB = {"contradiction": 0, "entailment": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP_RCB.items()}

# ─────────────────────── preprocess ────────────────────────────────
def preprocess_rcb(example, tok, max_len=512):
    p = example["premise"]    + tok.eos_token
    h = example["hypothesis"] + tok.eos_token

    enc = tok(p, h,
              add_special_tokens=False,
              max_length=max_len,
              truncation=True,
              padding=False)

    # строковую метку → число в поле **label**
    if "label" in example:
        enc["label"] = LABEL_MAP_RCB[example["label"]]
    return enc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main(
    tokenizer_dir: str,
    model_dir: str,
    rsglue_dir: str,
    output_dir: str
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    logger.info('tokenizer was loaded')
    cls_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=None)
    logger.info('cls_data_collator was defined')
    SEED = 42
    RSGLUE_DIR = rsglue_dir
    SAVE_DIR = output_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ---RCB---
    # logger.info('RCB')
    # rcb_raw_train = Dataset.from_json(RSGLUE_DIR + "RCB/train.jsonl")
    # rcb_raw_test = Dataset.from_json(RSGLUE_DIR + "RCB/test.jsonl")
    # rcb_raw_val = Dataset.from_json(RSGLUE_DIR + "RCB/val.jsonl")
    # logger.info('RCB data was loaded')
    # cols_to_drop = ['premise', 'hypothesis', 'verb', 'genre', 'idx']
    # rcb_train = rcb_raw_train.map(
    #     lambda x: preprocess_rcb(x, tokenizer), remove_columns=cols_to_drop
    # )
    # cols_to_drop = ['premise', 'hypothesis', 'verb', 'genre', 'idx']
    # rcb_val = rcb_raw_val.map(
    #     lambda x: preprocess_rcb(x, tokenizer), remove_columns=cols_to_drop
    # )
    # rcb_test = rcb_raw_test.map(
    #     lambda x: preprocess_rcb(x, tokenizer), remove_columns=cols_to_drop
    # )
    # logger.info('RCB data was processed')
    # seed_everything(SEED)
    # model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # training_args = TrainingArguments(
    #     output_dir=SAVE_DIR + "rcb_cp", # The output directory
    #     overwrite_output_dir=True,
    #     eval_strategy="epoch",
    #     num_train_epochs=10, # number of training epochs
    #     per_device_train_batch_size=8, # batch size for training
    #     per_device_eval_batch_size=8,  # batch size for evaluation
    #     learning_rate=1e-5,
    #     save_strategy='epoch',
    #     logging_steps = 5,
    #     fp16=(device.type != 'cpu'),
    #     weight_decay=0.01,
    #     push_to_hub=False,
    #     seed=42,
    #     load_best_model_at_end=True,
    #     metric_for_best_model='eval_loss',
    #     data_seed=42,
    #     save_total_limit=1,
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=cls_data_collator,
    #     train_dataset=rcb_train,
    #     eval_dataset=rcb_val,
    #     compute_metrics=compute_mc_accuracy,
    #     # prediction_loss_only=True,
    # )
    # trainer.train()
    # torch.cuda.empty_cache()
    # eval_accuracy = trainer.evaluate()['eval_accuracy']
    # logger.info(f"RCB eval accuracy is {eval_accuracy}")

    # # Прогнозирование на тестовых данных
    # predictions = trainer.predict(rcb_test).predictions
    # # Это кортеж вида (logits, some_extra_data)

    # logits = predictions[0]             # Берём только logits
    # print("Logits shape ->", logits.shape)

    # # Далее argmax по нужной оси, обычно axis=1 (если logits.shape=(N, num_labels)):
    # rcb_test_predict = np.argmax(logits, axis=1)
    
    # label_map_rcb = {0: 'contradiction' , 1: 'entailment', 2: 'neutral'}
    # rcb_test_predict = [
    #     {"idx":i, "label": label_map_rcb[rcb_test_predict[i]]} for i in range(rcb_test_predict.shape[0])
    # ]
    # with open(SAVE_DIR + 'RCB.jsonl', 'w') as f:
    #     for line in rcb_test_predict:
    #         f.write(f"{line}\n".replace("'", '"'))
    # del rcb_test_predict
    # del label_map_rcb
    # logger.info('RCB Done\n')





    # ------------------------- 2. подготовка датасетов -------------------------
    logger.info("PARus")

    parus_raw_train = Dataset.from_json(RSGLUE_DIR + "PARus/train.jsonl").map(prepreprocess_parus)
    parus_raw_val   = Dataset.from_json(RSGLUE_DIR + "PARus/val.jsonl").map(prepreprocess_parus)
    parus_raw_test  = Dataset.from_json(RSGLUE_DIR + "PARus/test.jsonl").map(prepreprocess_parus)

    tok_fn = lambda ex: preprocess_parus_binary(ex, tokenizer, max_len=512)

    parus_train = parus_raw_train.map(tok_fn, remove_columns=parus_raw_train.column_names)
    parus_val   = parus_raw_val.map(tok_fn,   remove_columns=parus_raw_val.column_names)
    parus_test  = parus_raw_test.map(tok_fn,  remove_columns=parus_raw_test.column_names)

    # ── фильтруем только те, где реально появились токены ──
    parus_train = parus_train.filter(lambda x: "input_ids" in x)
    parus_val   = parus_val.filter(lambda x: "input_ids" in x)
    parus_test  = parus_test.filter(lambda x: "input_ids" in x)

    from datasets import Value
    if "labels" in parus_train.column_names:
        parus_train = parus_train.cast_column("labels", Value("int64"))
        parus_val   = parus_val.cast_column("labels",   Value("int64"))

    logger.info(f"PARus processed ➜ train={len(parus_train)}, "
                f"val={len(parus_val)}, test={len(parus_test)}")


    # ------------------------- 3. Trainer (не менялся) -------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2,
        local_files_only=True,
    )
    model.config.problem_type = "single_label_classification"

    training_args = TrainingArguments(
        output_dir            = SAVE_DIR + "parus_cp",
        overwrite_output_dir  = True,
        evaluation_strategy   = "epoch",
        num_train_epochs      = 10,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        learning_rate         = 1e-5,
        fp16                  = (device.type != "cpu"),
        remove_unused_columns = False,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        data_collator   = DataCollatorWithPadding(tokenizer),
        train_dataset   = parus_train,
        eval_dataset    = parus_val,
        compute_metrics = compute_accuracy,
    )

    trainer.train()
    torch.cuda.empty_cache()

    # ------------------------- 4. инференс -------------------------
    logits = trainer.predict(parus_test).predictions[0]
    preds  = logits.argmax(-1)

    parus_test_pred = [{"idx": i, "label": int(preds[i])} for i in range(len(preds))]
    with open(SAVE_DIR + "PARus.jsonl", "w", encoding="utf-8") as f:
        for line in parus_test_pred:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info("PARus Done\n")



    #---TERRA---
    logger.info('TERRA')
    terra_raw_train = Dataset.from_json(RSGLUE_DIR + "TERRa/train.jsonl")
    terra_raw_test = Dataset.from_json(RSGLUE_DIR + "TERRa/test.jsonl")
    terra_raw_val = Dataset.from_json(RSGLUE_DIR + "TERRa/val.jsonl")
    logger.info('TERRA data was loaded')
    cols_to_drop = ['premise', 'hypothesis', 'idx']
    terra_train = (
        terra_raw_train.map(lambda x: preprocess_terra(x, tokenizer), remove_columns=cols_to_drop)
        .filter(lambda x: x is not None and len(x['input_ids']) > 0)
    )
    print(f'terra_train is {terra_train}')
    terra_val = terra_raw_val.map(
        lambda x: preprocess_terra(x, tokenizer), remove_columns=cols_to_drop
    ).filter(lambda x: x is not None and len(x['input_ids']) > 0)
    print(f'terra_val is {terra_val}')
    terra_test = terra_raw_test.map(
        lambda x: preprocess_terra(x, tokenizer), remove_columns=cols_to_drop
    ).filter(lambda x: x is not None and len(x['input_ids']) > 0)
    print(f'terra_test is {terra_test}')
    logger.info('TERRA data was processed')
    seed_everything(SEED)
    cls_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)


    base_collator = DataCollatorWithPadding(tokenizer)
    # # Уменьшите размер батча
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "terra_cp",  # Директория для сохранения результатов
        overwrite_output_dir=True,
        eval_strategy="epoch",
        num_train_epochs=10,  # Количество эпох
        per_device_train_batch_size=8,  # Уменьшение размера батча для обучения
        per_device_eval_batch_size=8,   # Уменьшение размера батча для оценки
        learning_rate=5e-6,
        save_strategy='epoch',
        logging_steps=5,
        fp16=(device.type != 'cpu'),
        weight_decay=0.01,
        push_to_hub=False,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        data_seed=42,
        save_total_limit=1,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=cls_data_collator,
        train_dataset=terra_train,
        eval_dataset=terra_val,
        compute_metrics=compute_accuracy,
    )


    trainer.train()

    # Очистка памяти
    torch.cuda.empty_cache()

    # Оценка
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"TERRA eval accuracy is {eval_accuracy}")

    # Прогнозирование на тестовых данных
    predictions = trainer.predict(terra_test).predictions
    # Это кортеж вида (logits, some_extra_data)

    logits = predictions[0]             # Берём только logits
    print("Logits shape ->", logits.shape)

    # Далее argmax по нужной оси, обычно axis=1 (если logits.shape=(N, num_labels)):
    terra_test_pred = np.argmax(logits, axis=1)

    label_map_terra = {0: 'entailment', 1: 'not_entailment'}
    terra_test_pred = [{"idx": i, "label": label_map_terra[terra_test_pred[i]]} for i in range(terra_test_pred.shape[0])]

    # Сохранение предсказаний в файл
    with open(SAVE_DIR + 'TERRa.jsonl', 'w') as f:
        for line in terra_test_pred:
            f.write(f"{line}\n".replace("'", '"'))
    logger.info('TERRa Done\n')






    # ------------------------------------------------------------------
    # LiDiRus
    # ------------------------------------------------------------------
    logger.info("LiDiRus")
    lidirus_raw_test = Dataset.from_json(RSGLUE_DIR + "LiDiRus/LiDiRus.jsonl")
    logger.info("LiDiRus data was loaded")

    # токенизируем и оставляем только нужные поля
    lidirus_test = lidirus_raw_test.map(
        lambda ex: preprocess_lidirus(ex, tokenizer),
        remove_columns=lidirus_raw_test.column_names,
    )

    # исключаем None-записи (на случай, если что-то пошло не так)
    lidirus_test = lidirus_test.filter(
        lambda x: x is not None and "input_ids" in x and x["input_ids"] is not None
    )

    logger.info("LiDiRus data was processed")

    # при инференсе лосс не нужен – убираем labels и говорим Trainer-у, что их нет
    if "labels" in lidirus_test.column_names:
        lidirus_test = lidirus_test.remove_columns("labels")
    trainer.label_names = []                     # <-- ключевая строка

    # предсказания
    pred_logits       = trainer.predict(lidirus_test).predictions[0]
    lidirus_test_pred = np.argmax(pred_logits, axis=-1)

    label_map_terra = {0: "entailment", 1: "not_entailment"}
    lidirus_test_pred = [
        {"idx": i, "label": label_map_terra[int(lidirus_test_pred[i])]}
        for i in range(lidirus_test_pred.shape[0])
    ]

    with open(SAVE_DIR + "LiDiRus.jsonl", "w", encoding="utf-8") as f:
        for line in lidirus_test_pred:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info("LiDiRus Done\n")




    # --- DaNetQA ---
    logger.info("DaNetQA")
    dnqa_raw_train = Dataset.from_json(RSGLUE_DIR + "DaNetQA/train.jsonl")
    dnqa_raw_val   = Dataset.from_json(RSGLUE_DIR + "DaNetQA/val.jsonl")
    dnqa_raw_test  = Dataset.from_json(RSGLUE_DIR + "DaNetQA/test.jsonl")
    logger.info("DaNetQA data was loaded")

    # токенизируем и сразу убираем старые колонки
    cols_to_drop = dnqa_raw_train.column_names          # все исходные поля
    dnqa_train = dnqa_raw_train.map(
        lambda x: preprocess_dnqa(x, tokenizer),
        remove_columns=cols_to_drop,
    )
    dnqa_val = dnqa_raw_val.map(
        lambda x: preprocess_dnqa(x, tokenizer),
        remove_columns=cols_to_drop,
    )
    cols_to_drop = [c for c in ["question", "passage", "idx", "label"]
                if c in dnqa_raw_test.column_names]          # ← фильтруем по наличию
    dnqa_test = dnqa_raw_test.map(
        lambda x: preprocess_dnqa(x, tokenizer),
        remove_columns=cols_to_drop,
    )

    # исключаем None-примеры
    dnqa_train = dnqa_train.filter(lambda x: x is not None)
    dnqa_val   = dnqa_val.filter(lambda x: x is not None)
    dnqa_test  = dnqa_test.filter(lambda x: x is not None)

    logger.info("DaNetQA data was processed")
    logger.info(f"Sample train: {dnqa_train[0]}")
    logger.info(f"Sample  val : {dnqa_val[0]}")

    seed_everything(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2,
        local_files_only=True,
    )

    base_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir            = SAVE_DIR + "danetqa_cp",
        overwrite_output_dir  = True,
        evaluation_strategy   = "epoch",
        num_train_epochs      = 10,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        learning_rate         = 1e-5,
        save_strategy         = "epoch",
        logging_steps         = 5,
        fp16                  = (device.type != "cpu"),
        weight_decay          = 0.01,
        seed                  = 42,
        load_best_model_at_end=True,
        metric_for_best_model ="eval_loss",
        data_seed             = 42,
        save_total_limit      = 1,
        remove_unused_columns = False,      # ← не выбрасывать input_ids!
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        data_collator   = base_collator,    # любой DataCollatorWithPadding(tokenizer)
        train_dataset   = dnqa_train,
        eval_dataset    = dnqa_val,
        compute_metrics = compute_accuracy,
    )

    trainer.train()
    torch.cuda.empty_cache()

    # ─────────────────────– инференс ──────────────────────
    eval_accuracy = trainer.evaluate()["eval_accuracy"]
    logger.info(f"DaNetQA eval accuracy = {eval_accuracy:.4f}")

    logits = trainer.predict(dnqa_test).predictions[0]
    dnqa_test_pred = np.argmax(logits, axis=1)

    label_map_dnqa = {0: "false", 1: "true"}
    dnqa_test_pred = [
        {"idx": i, "label": label_map_dnqa[int(p)]} for i, p in enumerate(dnqa_test_pred)
    ]

    with open(SAVE_DIR + "DaNetQA.jsonl", "w", encoding="utf-8") as f:
        for line in dnqa_test_pred:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info("DaNetQA Done\n")




    # ---RWSD---
    logger.info("RWSD")
    rwsd_raw_train = Dataset.from_json(RSGLUE_DIR + "RWSD/train.jsonl")
    rwsd_raw_val = Dataset.from_json(RSGLUE_DIR + "RWSD/val.jsonl")
    rwsd_raw_test = Dataset.from_json(RSGLUE_DIR + "RWSD/test.jsonl")

    cols_to_drop = ['idx', 'target', 'label', 'text']
    rwsd_train = rwsd_raw_train.map(lambda x: preprocess_rwsd(x, tokenizer), remove_columns=cols_to_drop)
    rwsd_val = rwsd_raw_val.map(lambda x: preprocess_rwsd(x, tokenizer), remove_columns=cols_to_drop)
    cols_to_drop = ['idx', 'target', 'text']
    rwsd_test = rwsd_raw_test.map(lambda x: preprocess_rwsd(x, tokenizer), remove_columns=cols_to_drop)
    logger.info('RWSD data was processed')
    seed_everything(SEED)
    base_collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2,
        local_files_only=True,
    )
    model.config.problem_type = "single_label_classification"   # ← теперь всегда CE-loss
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "rwsd_cp", # The output directory
        overwrite_output_dir=True,
        eval_strategy="epoch",
        num_train_epochs=10, # number of training epochs
        per_device_train_batch_size=8, # batch size for training
        per_device_eval_batch_size=8,  # batch size for evaluation
        learning_rate=1e-5,
        save_strategy='epoch',
        logging_steps = 5,
        fp16=(device.type != 'cpu'),
        weight_decay=0.01,
        push_to_hub=False,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        data_seed=42,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=base_collator,
        train_dataset=rwsd_train,
        eval_dataset=rwsd_val,
        compute_metrics=compute_accuracy,
        # prediction_loss_only=True,
    )
    trainer.train()
    torch.cuda.empty_cache()
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"RWSD eval accuracy is {eval_accuracy}")
   
   # Прогнозирование на тестовых данных
    predictions = trainer.predict(rwsd_test).predictions
    # Это кортеж вида (logits, some_extra_data)

    logits = predictions[0]             # Берём только logits
    print("Logits shape ->", logits.shape)

    # Далее argmax по нужной оси, обычно axis=1 (если logits.shape=(N, num_labels)):
    rwsd_test_pred = np.argmax(logits, axis=1)

    label_map_rwsd = {0: "False" , 1: "True"}
    rwsd_test_pred = [
        {"idx":i, "label": label_map_rwsd[rwsd_test_pred[i]]} for i in range(rwsd_test_pred.shape[0])
    ]
    with open(SAVE_DIR + 'RWSD.jsonl', 'w') as f:
        for line in rwsd_test_pred:
            f.write(f"{line}\n".replace("'", '"'))
    del rwsd_test_pred
    logger.info('RWSD Done\n')


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))
    logger.addHandler(c_handler)
    logger.setLevel(logging.INFO)

    tokenizer = '/userspace/tev/cache/local-fredt5-model'
    model = "/userspace/tev/cache/output/model_10k"
    rsglue = './combined/'
    output = '/userspace/tev/cache/output/preds/'

    main(tokenizer_dir=tokenizer, model_dir=model, rsglue_dir=rsglue, output_dir=output)



