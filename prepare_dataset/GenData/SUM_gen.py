import re
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings

warnings.filterwarnings("ignore")



# Предобработка текста
WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))


def sum_load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    cache_dir='/userspace/tev/cache/'

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    return model, tokenizer, device


def sum_gen_batch(articles, model, tokenizer, device):
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article) for article in articles],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        min_length=20,
        no_repeat_ngram_size=3,
        num_beams=5,
        length_penalty=1.0
    )

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]


@torch.no_grad()
def sum_process_dataset(input_path, output_path, batch_size):
    model, tokenizer, device = sum_load_model()
    updated_samples = []

    with open(input_path, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    batch_texts = []
    batch_samples = []

    with tqdm(total=total_lines, desc="SUM") as pbar:
        for line in lines:
            sample = json.loads(line)

            if sample.get("sum") is not None:
                updated_samples.append(sample)
                pbar.update(1)
                continue

            original_text = sample.get("text", "").strip()
            if not original_text:
                pbar.update(1)
                continue

            batch_texts.append(original_text)
            batch_samples.append(sample)

            if len(batch_texts) == batch_size:
                try:
                    batch_summaries = sum_gen_batch(batch_texts, model, tokenizer, device)
                    for i, summary in enumerate(batch_summaries):
                        batch_samples[i]["sum"] = summary
                        updated_samples.append(batch_samples[i])
                except Exception as e:
                    print(f"Error processing batch: {e}")

                batch_texts, batch_samples = [], []

            pbar.update(1)

        if batch_texts:
            try:
                batch_summaries = sum_gen_batch(batch_texts, model, tokenizer, device)
                for i, summary in enumerate(batch_summaries):
                    batch_samples[i]["sum"] = summary
                    updated_samples.append(batch_samples[i])
            except Exception as e:
                print(f"Error processing final batch: {e}")

            finally:
                torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"SUM complete. Processed {len(updated_samples)} samples.")


if __name__ == "__main__":
    input_file = "sample_data.jsonl"
    output_file = "sample_data_sum.jsonl"

    sum_process_dataset(input_file, output_file, batch_size=16)