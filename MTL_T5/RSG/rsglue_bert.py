from transformers import RobertaConfig, BertTokenizer, BertForSequenceClassification, RobertaForMultipleChoice, RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, DataCollatorWithPadding
from datasets import Dataset

import numpy as np
import logging
import argparse
import torch

from tools_ft import (
    DataCollatorForMultipleChoice,
    compute_mc_accuracy,
    compute_accuracy,
    seed_everything,
    preprocess_rcb,
    prepreprocess_parus,
    preprocess_parus,
    preprocess_terra,
    preprocess_lidirus,
    preprocess_dnqa,
    preprocess_rwsd,
    preprocess_russe
)


def main(
    tokenizer_dir: str,
    model_dir: str,
    rsglue_dir: str,
    output_dir: str
) -> None:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    logger.info('tokenizer was loaded')
    cls_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=None)
    logger.info('cls_data_collator was defined')
    SEED = 42
    RSGLUE_DIR = rsglue_dir
    SAVE_DIR = output_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # ---RCB---
    logger.info('RCB')
    rcb_raw_train = Dataset.from_json(RSGLUE_DIR + "RCB/train.jsonl")
    rcb_raw_test = Dataset.from_json(RSGLUE_DIR + "RCB/test.jsonl")
    rcb_raw_val = Dataset.from_json(RSGLUE_DIR + "RCB/val.jsonl")
    logger.info('RCB data was loaded')
    cols_to_drop = ['premise', 'hypothesis', 'verb', 'negation', 'genre', 'idx', 'no_negation']
    rcb_train = rcb_raw_train.map(
        lambda x: preprocess_rcb(x, tokenizer), remove_columns=cols_to_drop
    )
    cols_to_drop = ['premise', 'hypothesis', 'verb', 'negation', 'genre', 'idx']
    rcb_val = rcb_raw_val.map(
        lambda x: preprocess_rcb(x, tokenizer), remove_columns=cols_to_drop
    )
    rcb_test = rcb_raw_test.map(
        lambda x: preprocess_rcb(x, tokenizer), remove_columns=cols_to_drop
    )
    logger.info('RCB data was processed')
    seed_everything(SEED)
    model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "rcb_cp", # The output directory
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=10, # number of training epochs
        per_device_train_batch_size=64, # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        learning_rate=5e-5,
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
        data_collator=cls_data_collator,
        train_dataset=rcb_train,
        eval_dataset=rcb_val,
        compute_metrics=compute_mc_accuracy,
        # prediction_loss_only=True,
    )
    trainer.train()
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"RCB eval accuracy is {eval_accuracy}")
    rcb_test_predict = np.argmax(trainer.predict(rcb_test).predictions, axis=1)
    label_map_rcb = {0: 'contradiction' , 1: 'entailment', 2: 'neutral'}
    rcb_test_predict = [
        {"idx":i, "label": label_map_rcb[rcb_test_predict[i]]} for i in range(rcb_test_predict.shape[0])
    ]
    with open(SAVE_DIR + 'RCB.jsonl', 'w') as f:
        for line in rcb_test_predict:
            f.write(f"{line}\n".replace("'", '"'))
    del rcb_test_predict
    logger.info('RCB Done\n')

    # ---PARus---
    logger.info('PARus')
    prs_raw_train = Dataset.from_json(RSGLUE_DIR + "PARus/train.jsonl")
    prs_raw_test = Dataset.from_json(RSGLUE_DIR + "PARus/test.jsonl")
    prs_raw_val = Dataset.from_json(RSGLUE_DIR + "PARus/val.jsonl")
    logger.info('PARus data was loaded')
    cols_to_drop = ['premise', 'choice1', 'choice2', 'question', 'idx']
    prs_train = prs_raw_train.map(prepreprocess_parus).map(
        lambda x: preprocess_parus(x, tokenizer), remove_columns=cols_to_drop, batched=True
    )
    prs_val = prs_raw_val.map(prepreprocess_parus).map(
        lambda x: preprocess_parus(x, tokenizer), remove_columns=cols_to_drop, batched=True
    )
    prs_test = prs_raw_test.map(prepreprocess_parus).map(
        lambda x: preprocess_parus(x, tokenizer), remove_columns=cols_to_drop, batched=True
    )
    logger.info('PARus data was processed')
    seed_everything(SEED)
    model = RobertaForMultipleChoice.from_pretrained(model_dir)
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "parus_cp", # The output directory
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=10, # number of training epochs
        per_device_train_batch_size=32, # batch size for training
        per_device_eval_batch_size=32,  # batch size for evaluation
        learning_rate=5e-5,
        save_strategy='epoch',
        logging_steps = 5,
        fp16=(device.type != 'cpu'),
        weight_decay=0.01,
        push_to_hub=False,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        save_total_limit=1,
        data_seed=42
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prs_train,
        eval_dataset=prs_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"PARus eval accuracy is {eval_accuracy}")
    prs_test_predict = np.argmax(trainer.predict(prs_test).predictions, axis=1)
    prs_test_predict = [
        {"idx":i, "label": prs_test_predict[i]} for i in range(prs_test_predict.shape[0])
    ]
    with open(SAVE_DIR + 'PARus.jsonl', 'w') as f:
        for line in prs_test_predict:
            f.write(f"{line}\n".replace("'", '"'))
    logger.info('PARus Done\n')

    #---TERRA---
    logger.info('TERRA')
    terra_raw_train = Dataset.from_json(RSGLUE_DIR + "TERRa/train.jsonl")
    terra_raw_test = Dataset.from_json(RSGLUE_DIR + "TERRa/test.jsonl")
    terra_raw_val = Dataset.from_json(RSGLUE_DIR + "TERRa/val.jsonl")
    logger.info('TERRA data was loaded')
    cols_to_drop = ['premise', 'hypothesis', 'idx']
    terra_train = terra_raw_train.map(
        lambda x: preprocess_terra(x, tokenizer), remove_columns=cols_to_drop
    )
    terra_val = terra_raw_val.map(
        lambda x: preprocess_terra(x, tokenizer), remove_columns=cols_to_drop
    )
    terra_test = terra_raw_test.map(
        lambda x: preprocess_terra(x, tokenizer), remove_columns=cols_to_drop
    )
    logger.info('TERRA data was processed')
    seed_everything(SEED)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "terra_cp", # The output directory
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=10, # number of training epochs
        per_device_train_batch_size=64, # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        learning_rate=5e-5,
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
        data_collator=cls_data_collator,
        train_dataset=terra_train,
        eval_dataset=terra_val,
        compute_metrics=compute_accuracy,
        # prediction_loss_only=True,
    )
    trainer.train()
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"TERRA eval accuracy is {eval_accuracy}")
    terra_test_pred = np.argmax(trainer.predict(terra_test).predictions, axis=1)
    label_map_terra = {0: 'entailment' , 1: 'not_entailment'}
    terra_test_pred = [
        {"idx":i, "label": label_map_terra[terra_test_pred[i]]} for i in range(terra_test_pred.shape[0])
    ]
    with open(SAVE_DIR + 'TERRa.jsonl', 'w') as f:
        for line in terra_test_pred:
            f.write(f"{line}\n".replace("'", '"'))
    logger.info('TERRa Done\n')

    # ---LiDiRus---
    logger.info('LiDiRus')
    lidirus_raw_test = Dataset.from_json(RSGLUE_DIR + "LiDiRus/LiDiRus.jsonl")
    logger.info('LiDiRus data was loaded')
    cols_to_drop = ['idx', 'logic', 'predicate-argument-structure', 'lexical-semantics', 'knowledge']
    lidirus_test = lidirus_raw_test.map(
        lambda x: preprocess_lidirus(x, tokenizer), remove_columns=cols_to_drop
    )
    logger.info('LiDiRus data was processed')
    lidirus_test_pred = np.argmax(trainer.predict(lidirus_test).predictions, axis=1)
    label_map_terra = {0: 'entailment' , 1: 'not_entailment'}
    lidirus_test_pred = [
        {"idx": i, "label": label_map_terra[lidirus_test_pred[i]]} for i in range(lidirus_test_pred.shape[0])
    ]
    with open(SAVE_DIR + 'LiDiRus.jsonl', 'w') as f:
        for line in lidirus_test_pred:
            f.write(f"{line}\n".replace("'", '"'))
    logger.info('LiDiRus Done\n')

    # ---DaNetQA---
    logger.info('DaNetQA')
    dnqa_raw_train = Dataset.from_json(RSGLUE_DIR + "DaNetQA/train.jsonl")
    dnqa_raw_val = Dataset.from_json(RSGLUE_DIR + "DaNetQA/val.jsonl")
    dnqa_raw_test = Dataset.from_json(RSGLUE_DIR + "DaNetQA/test.jsonl")
    logger.info('DaNetQA data was loaded')

    cols_to_drop = ['question', 'passage', 'blabel']
    dnqa_train = dnqa_raw_train.map(
        lambda x: preprocess_dnqa(x, tokenizer), remove_columns=cols_to_drop
    )
    dnqa_val = dnqa_raw_val.map(
        lambda x: preprocess_dnqa(x, tokenizer), remove_columns=cols_to_drop
    )
    cols_to_drop = ['question', 'passage']
    dnqa_test = dnqa_raw_test.map(
        lambda x: preprocess_dnqa(x, tokenizer), remove_columns=cols_to_drop
    )
    logger.info('DaNetQA data was processed')
    seed_everything(SEED)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "danetqa_cp", # The output directory
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=10, # number of training epochs
        per_device_train_batch_size=16, # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation
        learning_rate=5e-5,
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
        data_collator=cls_data_collator,
        train_dataset=dnqa_train,
        eval_dataset=dnqa_val,
        compute_metrics=compute_accuracy,
        # prediction_loss_only=True,
    )
    trainer.train()
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"DaNetQA eval accuracy is {eval_accuracy}")
    dnqa_test_pred = np.argmax(trainer.predict(dnqa_test).predictions, axis=1)
    label_map_dnqa = {0: "false" , 1: "true"}
    dnqa_test_pred = [
        {"idx":i, "label": label_map_dnqa[dnqa_test_pred[i]]} for i in range(dnqa_test_pred.shape[0])
    ]
    with open(SAVE_DIR + 'DaNetQA.jsonl', 'w') as f:
        for line in dnqa_test_pred:
            f.write(f"{line}\n".replace("'", '"'))
    logger.info('DaNetQA Done\n')

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
    rwsd_train = rwsd_train.rename_column('ilabel', 'label')
    rwsd_val = rwsd_val.rename_column('ilabel', 'label')
    logger.info('RWSD data was processed')
    seed_everything(SEED)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "rwsd_cp", # The output directory
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=10, # number of training epochs
        per_device_train_batch_size=32, # batch size for training
        per_device_eval_batch_size=32,  # batch size for evaluation
        learning_rate=5e-5,
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
        data_collator=cls_data_collator,
        train_dataset=rwsd_train,
        eval_dataset=rwsd_val,
        compute_metrics=compute_accuracy,
        # prediction_loss_only=True,
    )
    trainer.train()
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"RWSD eval accuracy is {eval_accuracy}")
    rwsd_test_pred = np.argmax(trainer.predict(rwsd_test).predictions, axis=1)
    label_map_rwsd = {0: "False" , 1: "True"}
    rwsd_test_pred = [
        {"idx":i, "label": label_map_rwsd[rwsd_test_pred[i]]} for i in range(rwsd_test_pred.shape[0])
    ]
    with open(SAVE_DIR + 'RWSD.jsonl', 'w') as f:
        for line in rwsd_test_pred:
            f.write(f"{line}\n".replace("'", '"'))
    logger.info('RWSD Done\n')
    
    # ---RUSSE---
    logger.info("RUSSE")
    russ_raw_train = Dataset.from_json(RSGLUE_DIR + "RUSSE/train.jsonl")
    russ_raw_val = Dataset.from_json(RSGLUE_DIR + "RUSSE/val.jsonl")
    russ_raw_test = Dataset.from_json(RSGLUE_DIR + "RUSSE/test.jsonl")
    logger.info("RUSSE data was loaded")

    cols_to_drop = ['idx', 'word', 'sentence1', 'sentence2', 'start1', 'end1', 'start2', 'end2', 'label', 
                'gold_sense1', 'gold_sense2']
    russe_train = russ_raw_train.map(lambda x: preprocess_russe(x, tokenizer), remove_columns=cols_to_drop)
    russe_val = russ_raw_val.map(lambda x: preprocess_russe(x, tokenizer), remove_columns=cols_to_drop)
    cols_to_drop = ['idx', 'word', 'sentence1', 'sentence2', 'start1', 'end1', 'start2', 'end2']
    russe_test = russ_raw_test.map(lambda x: preprocess_russe(x, tokenizer), remove_columns=cols_to_drop)
    russe_train = russe_train.rename_column('ilabel', 'label')
    russe_val = russe_val.rename_column('ilabel', 'label')
    logger.info("RUSSE data was processed")
    seed_everything(42)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    training_args = TrainingArguments(
        output_dir=SAVE_DIR + "russe_cp", # The output directory
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=10, # number of training epochs
        per_device_train_batch_size=64, # batch size for training
        per_device_eval_batch_size=64,  # batch size for evaluation
        learning_rate=5e-5,
        save_strategy='epoch',
        logging_steps = 5,
        fp16=(device.type != 'cpu'),
        weight_decay=0.01,
        push_to_hub=False,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        data_seed=42,
        save_total_limit=1
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=cls_data_collator,
        train_dataset=russe_train,
        eval_dataset=russe_val,
        compute_metrics=compute_accuracy,
        # prediction_loss_only=True,
    )
    trainer.train()
    eval_accuracy = trainer.evaluate()['eval_accuracy']
    logger.info(f"RUSSE eval accuracy is {eval_accuracy}")
    russe_test_pred = np.argmax(trainer.predict(russe_test).predictions, axis=1)
    label_map_russe = {0: "false" , 1: "true"}
    russe_test_pred = [
        {"idx":i, "label": label_map_russe[russe_test_pred[i]]} for i in range(russe_test_pred.shape[0])
    ]
    with open(SAVE_DIR + 'RUSSE.jsonl', 'w') as f:
        for line in russe_test_pred:
            f.write(f"{line}\n".replace("'", '"'))
    logger.info('RUSSE Done\n')



if __name__ == '__main__':
    # Create looger
    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))
    logger.addHandler(c_handler)
    logger.setLevel(logging.INFO)

    # Parse args
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-t', '--tokenizer', required=True, type=str)
    argParser.add_argument('-m', '--model', required=True, type=str)
    argParser.add_argument('-r', '--rsglue', required=True, type=str)
    argParser.add_argument('-o', '--output', required=True, type=str)
    args = argParser.parse_args()

    main(tokenizer_dir=args.tokenizer, model_dir=args.model, rsglue_dir=args.rsglue, output_dir=args.output)
