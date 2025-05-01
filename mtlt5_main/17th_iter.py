# ------------- MUST be the first lines -------------
import os, pathlib
import json
import re
import torch
import os
import warnings
import logging
from typing import Literal, Dict, Optional
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
from rich.console import Console
from rich.logging import RichHandler
import matplotlib.pyplot as plt

from load_dataset import process_multitask_dataset
import random

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
# Для кэша HuggingFace
os.environ["HF_HOME"] = "/userspace/tev/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/userspace/tev/cache/"
os.environ["MPLCONFIGDIR"] = "/userspace/tev/cache/"


class FREDT5MultiTaskModel(torch.nn.Module):
    def __init__(self, model_params: dict) -> None:
        super(FREDT5MultiTaskModel, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.device == torch.device("cuda:0"):
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

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

        total_layers = len(self.base_model.encoder.block)
        split_at_layer = 6
        if split_at_layer >= total_layers:
            split_at_layer = total_layers - 1
        self.split_at_layer = split_at_layer

        self.encoder_first = self.base_model.encoder.block[:split_at_layer]
        self.encoder_second = self.base_model.encoder.block[split_at_layer:]

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Головные слои (NLU)
        self.nli_head = torch.nn.Linear(config.d_model, 3).to(self.device)
        self.mc_head = torch.nn.Linear(config.d_model, 17).to(self.device)
        self.ner_head = torch.nn.Linear(config.d_model, 29).to(self.device)

    def forward_first_encoder(self, input_ids, attention_mask):
        hidden_states = self.base_model.encoder.embed_tokens(input_ids)
        # hidden_states = hidden_states * (self.base_model.config.d_model ** 0.5)

        seq_len = hidden_states.size(1)
        cache_position = torch.arange(seq_len)

        dtype = hidden_states.dtype
        # Формируем 4D attention_mask
        attn_mask_4d = self._expand(attention_mask, dtype, tgt_len=seq_len)

        for layer in self.encoder_first:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                cache_position=cache_position,
                use_cache=False,
                past_key_value=None,
                output_attentions=False,
            )[0]

        return hidden_states

    def forward_second_encoder(self, hidden_states, attention_mask):
        seq_len = hidden_states.size(1)
        cache_position = torch.arange(seq_len)
        attn_mask_4d = self._expand(attention_mask, hidden_states.dtype, tgt_len=seq_len)

        for layer in self.encoder_second:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                cache_position=cache_position,
                use_cache=False,
                past_key_value=None,
                output_attentions=False,
            )[0]

        hidden_states = self.base_model.encoder.final_layer_norm(hidden_states)
        hidden_states = self.base_model.encoder.dropout(hidden_states)
        return hidden_states

    def forward_decoder(self,
                        hidden_states,  # encoder-вывод
                        attention_mask,
                        decoder_input_ids):  # обязательно Tensor, не None
        # 1. пропускаем через декодер
        dec_out = self.base_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        # 2. берём именно last_hidden_state
        seq_hidden = dec_out.last_hidden_state  # [B, T, d]

        # 3. layer-norm + dropout декодера
        seq_hidden = self.base_model.decoder.final_layer_norm(seq_hidden)
        seq_hidden = self.base_model.decoder.dropout(seq_hidden)

        # 4. lm-head
        return self.base_model.lm_head(seq_hidden)  # [B, T, vocab]

    def _expand(self, attn_mask, dtype, tgt_len):
        bsz, src_len = attn_mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        attn_mask = attn_mask.to(dtype)
        expanded = attn_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
        expanded = (1.0 - expanded) * torch.finfo(dtype).min
        return expanded

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                task_mode="gen"):
        task_mode = task_mode
        if task_mode == "gen":
            return self.forward_gen(input_ids, attention_mask, decoder_input_ids)
        elif task_mode == "nli":
            return self.forward_nli(input_ids, attention_mask, decoder_input_ids)
        elif task_mode == "mc":
            return self.forward_mc(input_ids, attention_mask, decoder_input_ids)
        elif task_mode == "ner":
            return self.forward_ner(input_ids, attention_mask, decoder_input_ids)
        else:
            ValueError(f"unknown {task_mode}")

    def forward_gen(self, input_ids, attention_mask=None, decoder_input_ids=None):
        hidden_states = self.forward_first_encoder(input_ids, attention_mask)
        hidden_states = self.forward_second_encoder(hidden_states, attention_mask)
        return self.forward_decoder(hidden_states, attention_mask, decoder_input_ids)

    def forward_nli(self, input_ids, attention_mask=None, decoder_input_ids=None):
        hidden_states = self.forward_first_encoder(input_ids, attention_mask)
        hidden_states = self.forward_second_encoder(hidden_states, attention_mask)  # You did 2 steps before
        pooled = hidden_states.mean(dim=1)
        return self.nli_head(pooled)

    def forward_mc(self, input_ids, attention_mask=None, decoder_input_ids=None):
        hidden_states = self.forward_first_encoder(input_ids, attention_mask)
        pooled = hidden_states.mean(dim=1)
        return self.mc_head(pooled)

    def forward_ner(self, input_ids, attention_mask=None, decoder_input_ids=None):
        hidden_states = self.forward_first_encoder(input_ids, attention_mask)
        # pooled = hidden_states.mean(dim=1)  # You didn't use it, and I am not sure
        return self.ner_head(hidden_states)


class DataSet:
    class Task:
        def __init__(self, name, prefix, mode, loss_fn):
            self.name = name
            self.prefix = prefix
            self.mode = mode
            self.loss_fn = loss_fn
            self.samples = []  # list of dict

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    def __init__(self):
        self.task_names = ["title", "paraphrase", "sum", "qg", "qa", "nli", "mc", "ner"]

        self.tasks = {}

        self.tasks["title"] = self.Task("title", "Дай заголовок тексту: ", "gen",
                                        torch.nn.CrossEntropyLoss(ignore_index=-100))
        self.tasks["paraphrase"] = self.Task("paraphrase", "Перефразируй текст: ", "gen",
                                             torch.nn.CrossEntropyLoss(ignore_index=-100))
        self.tasks["sum"] = self.Task("sum", "Напиши краткое изложение текста: ", "gen",
                                      torch.nn.CrossEntropyLoss(ignore_index=-100))
        self.tasks["qg"] = self.Task("qg", "Задай вопрос, ответ на который содежится в тексте: ", "gen",
                                     torch.nn.CrossEntropyLoss(ignore_index=-100))
        self.tasks["qa"] = self.Task("qa", "Ответь на вопрос согласно тексту: ", "gen",
                                     torch.nn.CrossEntropyLoss(ignore_index=-100))
        self.tasks["nli"] = self.Task("nli", "", "nli", torch.nn.CrossEntropyLoss(ignore_index=-100))
        self.tasks["mc"] = self.Task("mc", "", "mc", torch.nn.CrossEntropyLoss(ignore_index=-100))
        self.tasks["ner"] = self.Task("ner", "", "ner", torch.nn.CrossEntropyLoss(ignore_index=-100))

    @staticmethod
    def random_swap_indexes(length, n=1, frozen_prefix=0):
        indices = list(range(length))
        if length > frozen_prefix + 1:
            for _ in range(n):
                i, j = random.sample(range(frozen_prefix, length), 2)
                indices[i], indices[j] = indices[j], indices[i]
        return indices

    def augment(self, sample, prefix_words_count):
        if random.random() < 0.1:
            words = sample.split()
            new_order = self.random_swap_indexes(len(words), 1, prefix_words_count)
            words = [words[i] for i in new_order]
            sample = " ".join(words)
        return sample

    def add_tasks(self, inputs):
        for input in inputs:
            # logger.info(f'input is {input}')
            text = input.get("text", "").strip()
            for s in ["title", "paraphrase", "sum", "qg", "nli", "mc"]:  # all but "qa", "ner" - list of dicts
                task_str = input.get(s, "").strip()
                self.tasks[s].samples.append({"text": self.tasks[s].prefix + text, s: task_str})
            s = "qa"
            task_str = input.get(s, "").strip()
            question_str = input.get("qg", "").strip()
            self.tasks[s].samples.append({"text": self.tasks[s].prefix + text + "\n" + question_str, s: task_str})

            s = "ner"
            task_str = input.get(s, "")
            self.tasks[s].samples.append({"text": self.tasks[s].prefix + text, s: task_str})


ENC_MAX = 512
DEC_MAX = 384


class Trainer:
    def __init__(self, model: FREDT5MultiTaskModel, dataset: DataSet, dataset_val: DataSet, tokenizer):
        self.model = model
        self.dataset = dataset
        self.dataset_val = dataset_val
        self.tokenizer = tokenizer
        self.batch_size = 8

        self.output_dir = "/userspace/tev/cache/checkpoints"
        os.makedirs(self.output_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=model_params["LEARNING_RATE"],
            weight_decay=0.01
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=1e-6,
            max_lr=1e-4,
            step_size_up=20000,  # 1600000
            step_size_down=None,
            mode="exp_range",
            cycle_momentum=False,
            gamma=0.9,
        )

        self.ner_label_map = {
            "AGE": 0, "AWARD": 1, "CITY": 2, "COUNTRY": 3, "CRIME": 4, "DATE": 5,
            "DISEASE": 6, "DISTRICT": 7, "EVENT": 8, "FACILITY": 9, "FAMILY": 10,
            "IDEOLOGY": 11, "LANGUAGE": 12, "LAW": 13, "LOCATION": 14, "MONEY": 15,
            "NATIONALITY": 16, "NUMBER": 17, "ORDINAL": 18, "ORGANIZATION": 19,
            "PENALTY": 20, "PERCENT": 21, "PERSON": 22, "PRODUCT": 23,
            "PROFESSION": 24, "RELIGION": 25, "STATE_OR_PROVINCE": 26,
            "TIME": 27, "WORK_OF_ART": 28,
        }

        self.train_dataloader = {}
        self.train_dataloader_val = {}
        for name in self.dataset.tasks:
            task = self.dataset.tasks[name]
            self.train_dataloader[name] = DataLoader(
                task.samples,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda samples: samples
            )
            task = self.dataset_val.tasks[name]
            self.train_dataloader_val[name] = DataLoader(
                task.samples,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda samples: samples
            )

    def text_to_token(self, text: str, max_length: int = ENC_MAX) -> torch.Tensor:
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    def text_to_mask(self, text: str, max_length: int = ENC_MAX) -> torch.Tensor:
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).attention_mask.squeeze(0)

    def train(self, n_epoch):
        loss_avg = []
        val_loss_avg = []
        current_lr_hist = []
        for epoch in range(n_epoch):
            loss = self.train_epoch(self.train_dataloader, True)
            loss_avg.append(loss)
            val_loss = self.train_epoch(self.train_dataloader_val, False)
            val_loss_avg.append(val_loss)
            self.save_checkpoint(epoch, val_loss)

            # ---- 1) График average_train_loss ----
            plt.figure()
            plt.plot(range(0, epoch + 1), loss_avg, marker='o', label="Train Loss")
            plt.title("Average Train Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("/userspace/tev/cache/output/average_train_loss.png")
            plt.close()

            # ---- 2) График average_val_loss ----
            plt.figure()
            plt.plot(range(0, epoch + 1), val_loss_avg, marker='s', color="red", label="Val Loss")
            plt.title("Average Validation Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("/userspace/tev/cache/output/average_val_loss.png")
            plt.close()

            current_lr = self.optimizer.param_groups[0]["lr"]
            current_lr_hist.append(current_lr)

            plt.figure()
            plt.plot(range(0, epoch + 1), current_lr_hist, marker='s', color="green", label="Val Loss")
            plt.title("Current Learning Rate history")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.legend()
            plt.grid(True)
            plt.savefig("/userspace/tev/cache/output/current_lr_hist.png")
            plt.close()

    def calculate_average(self, data_dict):
        if not data_dict:  # пустой словарь
            return 0.0
        return sum(data_dict.values()) / len(data_dict)

    def train_epoch(self, train_dataloader, trainable):
        loss = {key: 0 for key in self.dataset.task_names}
        running_count = len(self.dataset.tasks)
        total_batch_count = 0
        accumulation_steps = 2

        # ── итераторы по каждому таску ───────────────────────
        data_iterator = {}
        for name in self.dataset.tasks:
            data_iterator[name] = iter(train_dataloader[name])
            total_batch_count += len(train_dataloader[name])

        tracker = tqdm(total=total_batch_count, desc=str(loss))
        step_count = 0

        epoch_total_loss = 0
        while running_count > 0:
            for name in self.dataset.tasks:
                try:
                    batch = next(data_iterator[name])
                    new_loss = self.train_batch(name, batch)  # tensor
                    (new_loss / accumulation_steps).backward()  # градиенты
                    loss[name] = new_loss.item()
                except StopIteration:
                    running_count -= 1
                    continue

                step_count += 1
                if step_count % accumulation_steps == 0:
                    if trainable:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                avg_loss = self.calculate_average(loss)
                tracker.set_description(f"{loss}  avg:{avg_loss:.4f}")
                tracker.update()
                epoch_total_loss += avg_loss
        tracker.set_description(f"{epoch_total_loss / step_count}")

        tracker.close()
        return epoch_total_loss / step_count

    def train_batch(self, name, batch):
        if name in ["title", "paraphrase", "sum", "qg", "qa"]:
            return self.train_batch_gen(batch, name, self.dataset.tasks[name].loss_fn)
        elif name == "nli":
            return self.train_batch_nli(batch, name, self.dataset.tasks[name].loss_fn)
        elif name == "mc":
            return self.train_batch_mc(batch, name, self.dataset.tasks[name].loss_fn)
        elif name == "ner":
            return self.train_batch_ner(batch, name, self.dataset.tasks[name].loss_fn)
        else:
            ValueError(f"unknown {name}")
            return torch.tensor(0)

    def train_batch_gen(self, batch, label_name, loss_fn):
        texts = [s["text"] for s in batch]
        labels = [s[label_name] for s in batch]

        prefix_words_count = len(self.dataset.tasks[label_name].prefix.split())
        for i, text in enumerate(texts):
            texts[i] = self.dataset.augment(text, prefix_words_count)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [self.text_to_token(t, ENC_MAX) for t in texts],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.model.device)

        attn_mask = torch.nn.utils.rnn.pad_sequence(
            [self.text_to_mask(t, ENC_MAX) for t in texts],
            batch_first=True,
            padding_value=0,
        ).to(self.model.device)

        label_ids = torch.nn.utils.rnn.pad_sequence(
            [self.text_to_token(t, DEC_MAX) for t in labels],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.model.device)

        # teacher-forcing: decoder получает target, сдвинутый на 1
        decoder_input_ids = label_ids[:, :-1]
        gold = label_ids[:, 1:]

        # --- прямой проход ---
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            task_mode="gen",
        )  # [B, T-1, vocab]

        # --- loss ---
        min_len = min(logits.size(1), gold.size(1))
        loss = loss_fn(
            logits[:, :min_len, :].reshape(-1, logits.size(-1)),
            gold[:, :min_len].reshape(-1),
        )
        return loss

    # ---------- NLI (3-классовая классификация) ----------
    def train_batch_nli(self, batch, label_name, loss_fn):
        texts = [s["text"] for s in batch]
        labels = [s[label_name] for s in batch]

        prefix_words_count = len(self.dataset.tasks[label_name].prefix.split())
        for i, text in enumerate(texts):
            texts[i] = self.dataset.augment(text, prefix_words_count)

        # входы
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [self.text_to_token(t, ENC_MAX) for t in texts],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.model.device)

        attn_mask = torch.nn.utils.rnn.pad_sequence(
            [self.text_to_mask(t, ENC_MAX) for t in texts],
            batch_first=True,
            padding_value=0,
        ).to(self.model.device)

        # метки
        label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
        label_ids = torch.tensor(
            [label_map[t] for t in labels],
            device=self.model.device,
        )

        # прямой проход + loss
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            task_mode="nli",
        )
        loss = loss_fn(logits, label_ids)
        return loss

    # ---------- MC (17 тематик) ----------
    def train_batch_mc(self, batch, label_name, loss_fn):
        # 1. разложить батч (list[dict]) на списки строк
        texts = [s["text"] for s in batch]  # list[str]
        labels = [s[label_name] for s in batch]  # list[str]

        prefix_words_count = len(self.dataset.tasks[label_name].prefix.split())
        for i, text in enumerate(texts):
            texts[i] = self.dataset.augment(text, prefix_words_count)

        # 2. токенизировать и падить
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [self.text_to_token(t, ENC_MAX) for t in texts],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.model.device)

        attn_mask = torch.nn.utils.rnn.pad_sequence(
            [self.text_to_mask(t, ENC_MAX) for t in texts],
            batch_first=True,
            padding_value=0,
        ).to(self.model.device)

        # 3. метки → tensor
        label_map = {
            'education': 0, 'human interest': 1, 'society': 2, 'sport': 3,
            'crime, law and justice': 4, 'disaster, accident and emergency incident': 5,
            'arts, culture, entertainment and media': 6, 'politics': 7,
            'economy, business and finance': 8, 'lifestyle and leisure': 9,
            'science and technology': 10, 'health': 11, 'labour': 12,
            'religion': 13, 'weather': 14, 'environment': 15,
            'conflict, war and peace': 16,
        }
        label_ids = torch.tensor(
            [label_map[x] for x in labels],
            device=self.model.device,
        )

        # 4. прямой проход
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            task_mode="mc",
        )  # [B, 17]

        # 5. loss
        loss = loss_fn(logits, label_ids)
        return loss

    def build_ner_labels(self, text: str, ner_items: list, tokenizer, label_map):
        """
        text        : исходная строка
        ner_items   : список словарей {'word', 'tag', 'index'}
        tokenizer   : токенизатор HF (T5/BPE/…)
        label_map   : {"ORG":19, ...}
        -------------------------------
        return: torch.LongTensor [seq_len] с id-ами тегов (или -100)
        """
        prefix_words_count = len(self.dataset.tasks["ner"].prefix.split())
        text = self.dataset.augment(text, prefix_words_count)
        # 1. Токенизируем с сохранением word_ids
        enc = tokenizer(
            text,
            truncation=True,
            padding=False,
            return_offsets_mapping=False,
            return_attention_mask=False,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=ENC_MAX
        )
        input_ids = enc.input_ids.squeeze(0)  # [seq_len]
        word_ids = enc.word_ids()  # list[Optional[int]]

        # 2. Готовим массив меток
        seq_len = len(input_ids)
        labels = torch.full((seq_len,), -100, dtype=torch.long)  # по-умолчанию «игнорировать»

        # 3. Быстрое сопоставление: word_index → tag_id
        idx2tag = {item["index"]: label_map[item["tag"]] for item in ner_items}

        # 4. Заполняем labels: для каждого токена узнаём, к какому слову он относится
        for pos, widx in enumerate(word_ids):
            if widx is None:  # спец-токен <pad>, </s>, etc.
                continue
            if widx in idx2tag:
                labels[pos] = idx2tag[widx]

        return input_ids, labels

    # ---------- NER (токен-классификация, 29 классов) ----------
    def train_batch_ner(self, batch, label_name, loss_fn):
        input_ids_lst, attn_masks, label_lst = [], [], []

        for sample in batch:
            ids, lbl = self.build_ner_labels(
                text=sample["text"],
                ner_items=sample[label_name],
                tokenizer=self.tokenizer,
                label_map=self.ner_label_map,  # заранее объявите dict
            )
            mask = torch.ones_like(ids)  # потому что без pad
            input_ids_lst.append(ids)
            attn_masks.append(mask)
            label_lst.append(lbl)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_lst, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.model.device)

        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_masks, batch_first=True, padding_value=0
        ).to(self.model.device)

        labels = torch.nn.utils.rnn.pad_sequence(
            label_lst, batch_first=True, padding_value=-100
        ).to(self.model.device)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            task_mode="ner",
        )  # [B, T, 29]

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint_path = os.path.join(
            self.output_dir, f"epoch_{epoch + 1}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved → {checkpoint_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs.txt", encoding="utf-8"),
                  RichHandler()]
    )

    logger.info('Обучение начато!')

    train_data = process_multitask_dataset("data/merged_data_f_sample_1_1_out_1_f.jsonl", 1)
    valid_data = process_multitask_dataset("data/merged_data_f_sample_1_1_out_2_f.jsonl", 1)

    # train_data = process_multitask_dataset("data/sample_data.jsonl", 1)
    # valid_data = process_multitask_dataset("data/sample_data.jsonl", 1)

    model_params = {
        "MODEL": "/userspace/tev/cache/local-fredt5-model",  # или путь к FRED-T5-1.7B
        "LEARNING_RATE": 1e-4
    }

    model = FREDT5MultiTaskModel(model_params=model_params)

    dataset = DataSet()
    dataset_val = DataSet()
    dataset.add_tasks(train_data)
    dataset_val.add_tasks(valid_data)

    trainer = Trainer(model, dataset, dataset_val, model.tokenizer)
    trainer.train(4)

    logger.info('Обучение завершено!')
