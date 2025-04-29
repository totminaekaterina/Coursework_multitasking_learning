# ------------- MUST be the first lines -------------
import os, pathlib
import json
import re
import torch
import os
import warnings
import logging
from typing import Literal, Dict, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
from rich.console import Console
from rich.logging import RichHandler
import matplotlib.pyplot as plt

from load_dataset import process_multitask_dataset
import random
warnings.filterwarnings("ignore")
# --------------------- МОДЕЛЬ ---------------------
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config

# Для кэша HuggingFace
os.environ["HF_HOME"] = "/userspace/tev/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/userspace/tev/cache/"
os.environ["MPLCONFIGDIR"] = "/userspace/tev/cache/"


class FREDT5MultiTaskModel(torch.nn.Module):
    def __init__(self, model_params: dict) -> None:
        super(FREDT5MultiTaskModel, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Загружаем модель
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_params["MODEL"],
            torch_dtype=torch.float32,
            use_cache=False
        )
        self.base_model.config.use_cache = False
        self.base_model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_params["MODEL"], 
            eos_token='</s>', 
            pad_token='<pad>'
        )
        config = self.base_model.config
        if not isinstance(config, T5Config):
            raise ValueError("Ожидается T5Config.")

        # Разделим энкодер на «нижние»/«верхние»
        total_layers = len(self.base_model.encoder.block)
        split_at_layer = 6
        if split_at_layer >= total_layers:
            split_at_layer = total_layers - 1
        self.split_at_layer = split_at_layer

        self.encoder_lower = self.base_model.encoder.block[:split_at_layer]
        self.encoder_upper = self.base_model.encoder.block[split_at_layer:]

        # Адаптер
        d_model = config.d_model
        self.adapter_proj = nn.Linear(d_model, d_model).to(self.device)

        # Головные слои (NLU)
        self.nli_head = nn.Linear(d_model, 3).to(self.device)
        self.mc_head  = nn.Linear(d_model, 17).to(self.device)
        self.ner_head = nn.Linear(d_model, 29).to(self.device)

        # --- CHANGED FOR STS ---
        self.sts_head = nn.Linear(d_model, 1).to(self.device)  # Выдаём скаляр (similarity score)

    def forward_lower_encoder(self, input_ids, attention_mask):
        hidden_states = self.base_model.encoder.embed_tokens(input_ids)
        hidden_states = hidden_states * (self.base_model.config.d_model ** 0.5)
        hidden_states = self.base_model.encoder.dropout(hidden_states)

        seq_len = hidden_states.size(1)
        cache_position = torch.arange(seq_len, device=hidden_states.device)

        dtype = hidden_states.dtype
        # Формируем 4D attention_mask
        attn_mask_4d = self._expand(attention_mask, dtype, tgt_len=seq_len)

        for layer in self.encoder_lower:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                cache_position=cache_position,
                use_cache=False,
                past_key_value=None,
                output_attentions=False,
            )[0]

        return hidden_states

    def forward_upper_encoder(self, hidden_states, attention_mask):
        hidden_states = self.adapter_proj(hidden_states)

        seq_len = hidden_states.size(1)
        cache_position = torch.arange(seq_len, device=hidden_states.device)

        dtype = hidden_states.dtype
        attn_mask_4d = self._expand(attention_mask, dtype, tgt_len=seq_len)

        for layer in self.encoder_upper:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                cache_position=cache_position,
                use_cache=False,
                past_key_value=None,
                output_attentions=False,
            )[0]
        return hidden_states

    def _expand(self, attn_mask, dtype, tgt_len):
        bsz, src_len = attn_mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        attn_mask = attn_mask.to(dtype)
        expanded = attn_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
        expanded = (1.0 - expanded) * torch.finfo(dtype).min
        return expanded

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                decoder_input_ids=None,
                task_prefix=None,
                sts_labels=None,
                text_input_ids=None,
                text_attention_mask=None,
                paraphrase_input_ids=None,
                paraphrase_attention_mask=None,
                nli_labels=None, 
                mc_labels=None,
                ner_labels=None):

        # Словарь лоссов
        loss_dict = {
            "nli": None, 
            "mc": None, 
            "ner": None, 
            "nlg": None,
            "sts": None
        }

        # 1) Нижние слои
        if task_prefix == "sts":
            pass
        else:
            # Обычная логика: lower encoder
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            hidden_lower = self.forward_lower_encoder(input_ids, attention_mask)

        # --- Обработка NER ---
        if ner_labels is not None:
            ner_labels = ner_labels.to(self.device)
            ner_logits = self.ner_head(hidden_lower)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_ner = loss_fct(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))
            loss_dict["ner"] = loss_ner

        # --- Обработка MC ---
        if mc_labels is not None:
            mc_labels = mc_labels.to(self.device)
            pooled = hidden_lower.mean(dim=1)
            mc_logits = self.mc_head(pooled)
            loss_fct = nn.CrossEntropyLoss()
            loss_mc = loss_fct(mc_logits, mc_labels.squeeze(-1))
            loss_dict["mc"] = loss_mc

        # --- NLI и NLG ---
        is_nli = (nli_labels is not None)
        is_nlg = (task_prefix in ["title", "paraphrase", "sum", "qg", "qa"])

        hidden_upper = None
        if is_nli or is_nlg:
            hidden_upper = self.forward_upper_encoder(hidden_lower, attention_mask)

        # --- NLI ---
        if nli_labels is not None and hidden_upper is not None:
            nli_labels = nli_labels.to(self.device)
            pooled = hidden_upper.mean(dim=1)
            logits_nli = self.nli_head(pooled)
            loss_fct = nn.CrossEntropyLoss()
            loss_nli = loss_fct(logits_nli, nli_labels.squeeze(-1))
            loss_dict["nli"] = loss_nli

        # --- NLG ---
        if is_nlg and hidden_upper is not None:
            # Декодер T5
            if labels is not None:
                labels = labels.to(self.device)
            if decoder_input_ids is not None and decoder_input_ids.numel() > 0:
                decoder_input_ids = decoder_input_ids.to(self.device)
            else:
                decoder_input_ids = torch.full(
                    (hidden_upper.size(0), 1),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.device
                )

            # Вызов встроенного декодера
            outputs = self.base_model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=None,
                encoder_hidden_states=hidden_upper,
                encoder_attention_mask=attention_mask,
                return_dict=True,
                use_cache=False
            )
            seq_hidden = outputs.last_hidden_state
            lm_logits = self.base_model.lm_head(seq_hidden)

            if labels is not None and labels.size(1) > 0:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                min_len = min(lm_logits.size(1), labels.size(1))
                if min_len > 0:
                    loss = loss_fct(
                        lm_logits[:, :min_len, :].contiguous().view(-1, lm_logits.size(-1)),
                        labels[:, :min_len].contiguous().view(-1)
                    )
                    loss_dict["nlg"] = loss

        # --- CHANGED FOR STS ---
        if task_prefix == "sts" and sts_labels is not None:
            text_input_ids = text_input_ids.to(self.device)
            text_attention_mask = text_attention_mask.to(self.device)
            para_input_ids = paraphrase_input_ids.to(self.device)
            para_attention_mask = paraphrase_attention_mask.to(self.device)

            # прогоняем text_input_ids через lower+upper
            hidden_lower_text = self.forward_lower_encoder(text_input_ids, text_attention_mask)
            hidden_upper_text = self.forward_upper_encoder(hidden_lower_text, text_attention_mask)

            # то же для paraphrase
            hidden_lower_para = self.forward_lower_encoder(para_input_ids, para_attention_mask)
            hidden_upper_para = self.forward_upper_encoder(hidden_lower_para, para_attention_mask)

            # pool
            text_vec = hidden_upper_text.mean(dim=1)
            para_vec = hidden_upper_para.mean(dim=1)
            # склеиваем
            # combined = torch.cat([text_vec, para_vec], dim=1)
            combined = text_vec + para_vec 

            # подаём в sts_head -> скаляр
            logits_sts = self.sts_head(combined)
            # MSELoss
            sts_labels = sts_labels.to(self.device)
            loss_fct = nn.MSELoss()
            loss_sts = loss_fct(logits_sts.squeeze(-1), sts_labels.float())
            loss_dict["sts"] = loss_sts

        return loss_dict



class T5Trainer:
    """
    Класс обучения. 
    Убрано деление на stage, чтобы модель обучалась одновременно на всех задачах.
    """
    def __init__(
        self, 
        model: FREDT5MultiTaskModel,
        tokenizer,
        model_params: dict,
        train_data,
        valid_data,
        output_dir: str = None,
        patience: int = 3,
        min_delta: float = 0.01
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.device = model.device
        self.output_dir = output_dir or "./output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=model_params["LEARNING_RATE"]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

        self.logger = logging.getLogger(__name__)
        # Параметры ранней остановки
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False

        # Задаём веса для задач ("больше внимания" сложным)
        self.task_weights = {
            "ner": 1.0,
            "mc": 1.0,
            "nli": 1.5,
            "nlg": 2.0
        }

    def random_swap(self, words, n=1):
        """
        Nеняем местами n пар слов. Если слов меньше 2, возвращаем без изменений.
        """
        if len(words) < 2:
            return words
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return words

    def preprocess_data(self, example):
        """
        Добавлен блок, который с вероятностью 15% зашумляет (swap-ауга) исходный text
        """
        outputs = []
        input_text = example.get("text", "").strip()
        if not input_text:
            self.logger.warning("Пропущен пример с пустым текстом")
            return outputs

        # --- 15% аугментация ---
        if random.random() < 0.15:
            words = input_text.split()
            # Один swap
            words = self.random_swap(words, n=1)
            input_text = " ".join(words)

        def prepare_generation_task(task_prefix, in_text, target_text):
            model_inputs = self.tokenizer(
                in_text,
                max_length=512,
                padding="max_length",
                truncation=True
            )
            labels_enc = self.tokenizer(
                target_text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()

            if labels_enc.numel() == 0:
                self.logger.warning("Пустой labels после токенизации для генеративной задачи")
                return None

            model_inputs["labels"] = labels_enc
            model_inputs["task_prefix"] = task_prefix
            return model_inputs

        # --- TITLE ---
        if "title" in example and example["title"].strip():
            inputs = f"title: {input_text}"
            targets = example["title"].strip()
            task_inputs = prepare_generation_task("title", inputs, targets)
            if task_inputs:
                outputs.append(task_inputs)

        # --- PARAPHRASE ---
        if "paraphrase" in example and example["paraphrase"]:
            inputs = f"paraphrase: {example['text']}"
            targets = example["paraphrase"]
            task_inputs = prepare_generation_task("paraphrase", inputs, targets)
            if task_inputs:
                outputs.append(task_inputs)

        # --- SUM ---
        if "sum" in example and example["sum"]:
            inputs = f"sum: {example['text']}"
            targets = example["sum"]
            task_inputs = prepare_generation_task("sum", inputs, targets)
            if task_inputs:
                outputs.append(task_inputs)

        # --- QG ---
        if "qg" in example and example["qg"]:
            inputs = f"qg: {example['text']}"
            targets = example["qg"]
            task_inputs = prepare_generation_task("qg", inputs, targets)
            if task_inputs:
                outputs.append(task_inputs)

        # --- QA ---
        if "qa" in example and example["qa"]:
            inputs = f"qa: {example['text']}"
            targets = example["qa"]
            task_inputs = prepare_generation_task("qa", inputs, targets)
            if task_inputs:
                outputs.append(task_inputs)

        # --- NLI ---
        if "nli" in example:
            label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
            if example["nli"] not in label_map:
                raise ValueError(f"Invalid NLI label: {example['nli']}")
            inputs = f"NLI: {input_text}"
            model_inputs = self.tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
            model_inputs["labels"] = torch.tensor([label_map[example["nli"]]], dtype=torch.long)
            model_inputs["task_prefix"] = "nli"
            outputs.append(model_inputs)

        # --- MC ---
        if "mc" in example and example["mc"]:
            task_prefix = "mc"
            inputs = f"{task_prefix}: {input_text}"
            model_inputs = self.tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

            mc_map = {
                'education': 0, 'human interest': 1, 'society': 2, 'sport': 3, 
                'crime, law and justice': 4, 'disaster, accident and emergency incident': 5,
                'arts, culture, entertainment and media': 6, 'politics': 7, 
                'economy, business and finance': 8, 'lifestyle and leisure': 9,
                'science and technology': 10, 'health': 11, 'labour': 12, 
                'religion': 13, 'weather': 14, 'environment': 15, 
                'conflict, war and peace': 16
            }
            if example["mc"] not in mc_map:
                raise ValueError(f"Invalid MC label: {example['mc']}")

            label = mc_map[example["mc"]]
            model_inputs["labels"] = torch.tensor([label], dtype=torch.long)
            model_inputs["task_prefix"] = "mc"
            outputs.append(model_inputs)

        # --- NER ---
        if "ner" in example:
            inputs = f"{input_text}"
            model_inputs = self.tokenizer(
                inputs,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True
            )
            offsets = model_inputs.pop("offset_mapping")
            text_ = example["text"]
            ner_labels = [-100] * len(model_inputs["input_ids"])
            tag_map = {
                "AGE": 0,"AWARD": 1,"CITY": 2,"COUNTRY": 3,"CRIME": 4,"DATE": 5,"DISEASE": 6,
                "DISTRICT": 7,"EVENT": 8,"FACILITY": 9,"FAMILY": 10,"IDEOLOGY": 11,"LANGUAGE": 12,
                "LAW": 13,"LOCATION": 14,"MONEY": 15,"NATIONALITY": 16,"NUMBER": 17,"ORDINAL": 18,
                "ORGANIZATION": 19,"PENALTY": 20,"PERCENT": 21,"PERSON": 22,"PRODUCT": 23,
                "PROFESSION": 24,"RELIGION": 25,"STATE_OR_PROVINCE": 26,"TIME": 27,"WORK_OF_ART": 28,
            }
            for entity in example["ner"]:
                word = entity["word"]
                tag  = entity["tag"]
                start_idx = text_.find(word)
                if start_idx == -1:
                    self.logger.warning(f"Entity '{word}' not found in text.")
                    continue
                end_idx = start_idx + len(word)
                for i, (ofs_start, ofs_end) in enumerate(offsets):
                    if ofs_start >= start_idx and ofs_end <= end_idx:
                        ner_labels[i] = tag_map.get(tag, -100)

            if all(lbl == -100 for lbl in ner_labels):
                self.logger.warning("Пропущен NER-пример без реальных меток.")
                return outputs
            model_inputs["labels"] = torch.tensor(ner_labels, dtype=torch.long)
            model_inputs["task_prefix"] = "ner"
            outputs.append(model_inputs)

        # --- STS ---
        if "sts" in example:
            score = float(example["sts"])
            text_tokens = self.tokenizer(
                example["text"],
                max_length=512,
                padding="max_length",
                truncation=True
            )
            paraphrase_tokens = self.tokenizer(
                example["paraphrase"],
                max_length=512,
                padding="max_length",
                truncation=True
            )
            model_inputs = {
                "text_input_ids": text_tokens["input_ids"],
                "text_attention_mask": text_tokens["attention_mask"],
                "paraphrase_input_ids": paraphrase_tokens["input_ids"],
                "paraphrase_attention_mask": paraphrase_tokens["attention_mask"],
                "labels": torch.tensor([score], dtype=torch.float32),
                "task_prefix": "sts"
            }
            outputs.append(model_inputs)

        return outputs

    def collate_fn(self, batch):
        # Генеративные
        gen_input_ids_list = []
        gen_attention_mask_list = []
        gen_labels_list = []

        # NLU
        input_ids_list = []
        attention_mask_list = []
        nli_labels_list = []
        mc_labels_list  = []
        ner_labels_list = []


        # --- добавляем STS ---
        sts_labels_list = []
        sts_text_input_ids_list = []
        sts_text_attention_mask_list = []
        sts_para_input_ids_list = []
        sts_para_attention_mask_list = []

        task_prefixes_list = []

        for item in batch:
            prefix = item['task_prefix']
            task_prefixes_list.append(prefix)

            if prefix in ["title", "paraphrase", "sum", "qa", "qg"]:
                # Генеративная задача
                gen_input_ids_list.append(torch.tensor(item['input_ids'], dtype=torch.long))
                gen_attention_mask_list.append(torch.tensor(item['attention_mask'], dtype=torch.long))
                gen_labels_list.append(torch.tensor(item['labels'], dtype=torch.long))

            elif prefix == 'nli':
                input_ids_list.append(torch.tensor(item['input_ids'], dtype=torch.long))
                attention_mask_list.append(torch.tensor(item['attention_mask'], dtype=torch.long))
                nli_labels_list.append(torch.tensor(item['labels'], dtype=torch.long))

            elif prefix == 'mc':
                input_ids_list.append(torch.tensor(item['input_ids'], dtype=torch.long))
                attention_mask_list.append(torch.tensor(item['attention_mask'], dtype=torch.long))
                mc_labels_list.append(torch.tensor(item['labels'], dtype=torch.long))

            elif prefix == 'ner':
                input_ids_list.append(torch.tensor(item['input_ids'], dtype=torch.long))
                attention_mask_list.append(torch.tensor(item['attention_mask'], dtype=torch.long))
                ner_labels_list.append(torch.tensor(item['labels'], dtype=torch.long))

            elif prefix == 'sts':
            # Именно тут собираем STS поля
                sts_labels_list.append(torch.tensor(item['labels'], dtype=torch.float))
                sts_text_input_ids_list.append(torch.tensor(item['text_input_ids'], dtype=torch.long))
                sts_text_attention_mask_list.append(torch.tensor(item['text_attention_mask'], dtype=torch.long))
                sts_para_input_ids_list.append(torch.tensor(item['paraphrase_input_ids'], dtype=torch.long))
                sts_para_attention_mask_list.append(torch.tensor(item['paraphrase_attention_mask'], dtype=torch.long))

        # Паддинг для генеративных
        if len(gen_input_ids_list) > 0:
            gen_input_ids = torch.nn.utils.rnn.pad_sequence(
                gen_input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            gen_attention_mask = torch.nn.utils.rnn.pad_sequence(
                gen_attention_mask_list, batch_first=True, padding_value=0
            )
            gen_labels = torch.nn.utils.rnn.pad_sequence(
                gen_labels_list, batch_first=True, padding_value=-100
            )
        else:
            gen_input_ids = None
            gen_attention_mask = None
            gen_labels = None

        # Паддинг для NLU
        if len(input_ids_list) > 0:
            input_ids = nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            attention_mask = nn.utils.rnn.pad_sequence(
                attention_mask_list, batch_first=True, padding_value=0
            )
        else:
            input_ids = None
            attention_mask = None

        def pad_or_none(lst, pad_value=-100, is_class=False):
            if len(lst) > 0:
                if is_class:
                    return torch.stack(lst, dim=0)
                else:
                    return nn.utils.rnn.pad_sequence(lst, batch_first=True, padding_value=pad_value)
            return None

        nli_labels = pad_or_none(nli_labels_list, is_class=True)
        mc_labels  = pad_or_none(mc_labels_list,  is_class=True)
        ner_labels = pad_or_none(ner_labels_list, pad_value=-100)

        # --- Паддим STS ---
        if len(sts_labels_list) > 0:
            sts_labels = torch.stack(sts_labels_list, dim=0)  # [B]
            text_input_ids = nn.utils.rnn.pad_sequence(
                sts_text_input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            text_attention_mask = nn.utils.rnn.pad_sequence(
                sts_text_attention_mask_list, batch_first=True, padding_value=0
            )
            paraphrase_input_ids = nn.utils.rnn.pad_sequence(
                sts_para_input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            paraphrase_attention_mask = nn.utils.rnn.pad_sequence(
                sts_para_attention_mask_list, batch_first=True, padding_value=0
            )
        else:
            sts_labels = None
            text_input_ids = None
            text_attention_mask = None
            paraphrase_input_ids = None
            paraphrase_attention_mask = None

        batch_out = {
            'task_prefix': task_prefixes_list,

            # Генеративные
            'gen_input_ids': gen_input_ids,
            'gen_attention_mask': gen_attention_mask,
            'gen_labels': gen_labels,

            # Общие NLU
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'nli_labels': nli_labels,
            'mc_labels': mc_labels,
            'ner_labels': ner_labels,

            # STS
            'sts_labels': sts_labels,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'paraphrase_input_ids': paraphrase_input_ids,
            'paraphrase_attention_mask': paraphrase_attention_mask,
        }
        return batch_out



    def train(self, batch_size: int = 1, epochs: int = 3):
        """
        одновременное мультитаск-обучение
        """
        train_data_processed = []
        for item in self.train_data:
            processed = self.preprocess_data(item)
            train_data_processed.extend(processed)

        valid_data_processed = []
        for item in self.valid_data:
            processed = self.preprocess_data(item)
            valid_data_processed.extend(processed)

        train_dataloader = DataLoader(
            train_data_processed,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_data_processed,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True
        )

        average_train_loss = []
        average_val_loss   = []
        accumulation_steps = 4

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            step_count = 0

            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for step, batch in enumerate(pbar):
                    ptr_gen = 0   # для title, paraphrase, sum, qg, qa
                    ptr_nli = 0
                    ptr_mc  = 0
                    ptr_ner = 0
                    ptr_sts = 0

                    total_batch_loss = torch.tensor(0.0, device=self.device)

                    # Проходим по всем элементам в batch["task_prefix"]
                    for i, prefix in enumerate(batch["task_prefix"]):
                        # prefix = "title", "nli", "mc", "ner" или "sts" и т.п.

                        if prefix in ["title", "paraphrase", "sum", "qa", "qg"]:
                            # См. блок генеративных
                            if batch["gen_labels"] is None:
                                continue
                            # Используем ptr_gen, а не i!
                            input_ids_  = batch["gen_input_ids"][ptr_gen].unsqueeze(0)
                            attn_mask_  = batch["gen_attention_mask"][ptr_gen].unsqueeze(0)
                            labels_     = batch["gen_labels"][ptr_gen].unsqueeze(0)
                            ptr_gen += 1

                            out = self.model(
                                input_ids=input_ids_,
                                attention_mask=attn_mask_,
                                labels=labels_,
                                task_prefix=prefix
                            )
                            loss_nlg = out["nlg"]
                            if loss_nlg is not None:
                                w = self.task_weights["nlg"]
                                total_batch_loss += w * (loss_nlg / accumulation_steps)

                        elif prefix == "nli":
                            if batch["nli_labels"] is None:
                                continue
                            input_ids_ = batch["input_ids"][ptr_nli].unsqueeze(0)
                            attn_mask_ = batch["attention_mask"][ptr_nli].unsqueeze(0)
                            labels_    = batch["nli_labels"][ptr_nli].unsqueeze(0)
                            ptr_nli += 1

                            out = self.model(
                                input_ids=input_ids_,
                                attention_mask=attn_mask_,
                                nli_labels=labels_,
                                task_prefix="nli"
                            )
                            loss_nli = out["nli"]
                            if loss_nli is not None:
                                w = self.task_weights["nli"]
                                total_batch_loss += w * (loss_nli / accumulation_steps)

                        elif prefix == "mc":
                            if batch["mc_labels"] is None:
                                continue
                            input_ids_ = batch["input_ids"][ptr_mc].unsqueeze(0)
                            attn_mask_ = batch["attention_mask"][ptr_mc].unsqueeze(0)
                            labels_    = batch["mc_labels"][ptr_mc].unsqueeze(0)
                            ptr_mc += 1

                            out = self.model(
                                input_ids=input_ids_,
                                attention_mask=attn_mask_,
                                mc_labels=labels_,
                                task_prefix="mc"
                            )
                            loss_mc = out["mc"]
                            if loss_mc is not None:
                                w = self.task_weights["mc"]
                                total_batch_loss += w * (loss_mc / accumulation_steps)

                        elif prefix == "sts":
                            if batch["sts_labels"] is None:
                                continue
                            labels_ = batch["sts_labels"][ptr_sts].unsqueeze(0)

                            out = self.model(
                                task_prefix="sts",
                                sts_labels=labels_,
                                text_input_ids=batch["text_input_ids"][ptr_sts].unsqueeze(0),
                                text_attention_mask=batch["text_attention_mask"][ptr_sts].unsqueeze(0),
                                paraphrase_input_ids=batch["paraphrase_input_ids"][ptr_sts].unsqueeze(0),
                                paraphrase_attention_mask=batch["paraphrase_attention_mask"][ptr_sts].unsqueeze(0)
                            )
                            ptr_sts += 1

                            loss_sts = out["sts"]
                            if loss_sts is not None:
                                w = 1.5
                                total_batch_loss += w * (loss_sts / accumulation_steps)

                        elif prefix == "ner":
                            if batch["ner_labels"] is None:
                                continue
                            input_ids_ = batch["input_ids"][ptr_ner].unsqueeze(0)
                            attn_mask_ = batch["attention_mask"][ptr_ner].unsqueeze(0)
                            labels_    = batch["ner_labels"][ptr_ner].unsqueeze(0)
                            ptr_ner += 1

                            out = self.model(
                                input_ids=input_ids_,
                                attention_mask=attn_mask_,
                                ner_labels=labels_,
                                task_prefix="ner"
                            )
                            loss_ner = out["ner"]
                            if loss_ner is not None:
                                w = self.task_weights["ner"]
                                total_batch_loss += w * (loss_ner / accumulation_steps)

                    # backward
                    if total_batch_loss.item() != 0.0:
                        total_batch_loss.backward()
                        step_count += 1
                        if step_count % accumulation_steps == 0 or (step+1) == len(train_dataloader):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            torch.cuda.empty_cache()

                    # Суммируем train loss
                    total_train_loss += total_batch_loss.item() * accumulation_steps

                    pbar.set_postfix({"Train Loss": f"{total_batch_loss.item():.4f}"})

            if len(train_dataloader) > 0:
                avg_train_loss = total_train_loss / len(train_dataloader)
            else:
                avg_train_loss = 0.0
            average_train_loss.append(avg_train_loss)

            # Валидация
            avg_val_loss = self.validate(valid_dataloader)
            average_val_loss.append(avg_val_loss)

            # Лог
            self._save_checkpoint(epoch, avg_val_loss)
            self.logger.info(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # ---- 1) График average_train_loss ----
        plt.figure()
        plt.plot(range(1, epochs+1), average_train_loss, marker='o', label="Train Loss")
        plt.title("Average Train Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("/userspace/tev/cache/output/average_train_loss.png")
        plt.close()

        # ---- 2) График average_val_loss ----
        plt.figure()
        plt.plot(range(1, epochs+1), average_val_loss, marker='s', color="red", label="Val Loss")
        plt.title("Average Validation Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("/userspace/tev/cache/output/average_val_loss.png")
        plt.close()

        return average_train_loss, average_val_loss


    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        step_cnt = 0

        with torch.no_grad():
            for batch in dataloader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                ptr_gen = 0
                ptr_nli = 0
                ptr_mc  = 0
                ptr_ner = 0
                ptr_sts = 0

                batch_loss = 0.0

                # Проходим по каждому элементу prefix
                for i, task_prefix in enumerate(batch["task_prefix"]):

                    # 1) Генеративные
                    if task_prefix in ["title", "paraphrase", "sum", "qa", "qg"]:
                        if batch["gen_labels"] is None:
                            continue
                        input_ids_  = batch["gen_input_ids"][ptr_gen].unsqueeze(0)
                        attn_mask_  = batch["gen_attention_mask"][ptr_gen].unsqueeze(0)
                        labels_     = batch["gen_labels"][ptr_gen].unsqueeze(0)
                        ptr_gen += 1

                        out = self.model(
                            input_ids=input_ids_,
                            attention_mask=attn_mask_,
                            labels=labels_,
                            task_prefix=task_prefix
                        )
                        if out["nlg"] is not None:
                            batch_loss += out["nlg"].item()

                    # 2) NLI
                    elif task_prefix == "nli":
                        if batch["nli_labels"] is None:
                            continue
                        input_ids_ = batch["input_ids"][ptr_nli].unsqueeze(0)
                        attn_mask_ = batch["attention_mask"][ptr_nli].unsqueeze(0)
                        labels_    = batch["nli_labels"][ptr_nli].unsqueeze(0)
                        ptr_nli += 1

                        out = self.model(
                            input_ids=input_ids_,
                            attention_mask=attn_mask_,
                            nli_labels=labels_,
                            task_prefix="nli"
                        )
                        if out["nli"] is not None:
                            batch_loss += out["nli"].item()

                    # 3) STS
                    elif task_prefix == "sts":
                        if batch["sts_labels"] is None:
                            continue
                        labels_ = batch["sts_labels"][ptr_sts].unsqueeze(0)
                        out = self.model(
                            task_prefix="sts",
                            sts_labels=labels_,
                            text_input_ids=batch["text_input_ids"][ptr_sts].unsqueeze(0),
                            text_attention_mask=batch["text_attention_mask"][ptr_sts].unsqueeze(0),
                            paraphrase_input_ids=batch["paraphrase_input_ids"][ptr_sts].unsqueeze(0),
                            paraphrase_attention_mask=batch["paraphrase_attention_mask"][ptr_sts].unsqueeze(0)
                        )
                        ptr_sts += 1

                        if out["sts"] is not None:
                            batch_loss += out["sts"].item()

                    # 4) MC
                    elif task_prefix == "mc":
                        if batch["mc_labels"] is None:
                            continue
                        input_ids_ = batch["input_ids"][ptr_mc].unsqueeze(0)
                        attn_mask_ = batch["attention_mask"][ptr_mc].unsqueeze(0)
                        labels_    = batch["mc_labels"][ptr_mc].unsqueeze(0)
                        ptr_mc += 1

                        out = self.model(
                            input_ids=input_ids_,
                            attention_mask=attn_mask_,
                            mc_labels=labels_,
                            task_prefix="mc"
                        )
                        if out["mc"] is not None:
                            batch_loss += out["mc"].item()

                    # 5) NER
                    elif task_prefix == "ner":
                        if batch["ner_labels"] is None:
                            continue
                        input_ids_ = batch["input_ids"][ptr_ner].unsqueeze(0)
                        attn_mask_ = batch["attention_mask"][ptr_ner].unsqueeze(0)
                        labels_    = batch["ner_labels"][ptr_ner].unsqueeze(0)
                        ptr_ner += 1

                        out = self.model(
                            input_ids=input_ids_,
                            attention_mask=attn_mask_,
                            ner_labels=labels_,
                            task_prefix="ner"
                        )
                        if out["ner"] is not None:
                            batch_loss += out["ner"].item()

                if batch_loss != 0:
                    total_loss += batch_loss
                    step_cnt += 1

        if step_cnt > 0:
            return total_loss / step_cnt
        else:
            return 0.0


    def _save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(self.output_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs.txt", encoding="utf-8"),
                  RichHandler()]
    )
    logger = logging.getLogger(__name__)

    train_data = process_multitask_dataset("data/merged_data_f_sample_1_1_out_1_f.jsonl", 1)
    valid_data = process_multitask_dataset("data/merged_data_f_sample_1_1_out_2_f.jsonl", 1)

    # train_data = process_multitask_dataset("sample_data.jsonl", 1)
    # valid_data = process_multitask_dataset("sample_data.jsonl", 1)

    model_params = {
        "MODEL": "/userspace/tev/cache/local-fredt5-model",  # или путь к FRED-T5-1.7B
        "LEARNING_RATE": 1e-4
    }


    model = FREDT5MultiTaskModel(model_params=model_params)
    trainer = T5Trainer(
        model=model,
        tokenizer=model.tokenizer,
        model_params=model_params,
        train_data=train_data,
        valid_data=valid_data,
        output_dir="/userspace/tev/cache/output",
        patience=3,
        min_delta=0.01
    )

    trainer.train(batch_size=8, epochs=5)
    logger.info('Обучение завершено!')
