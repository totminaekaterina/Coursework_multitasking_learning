import json
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def qd_load_model():
    cache_dir='/userspace/tev/cache/'

    tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base-multitask", cache_dir=cache_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base-multitask", cache_dir=cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def generate_batch(texts, model, tokenizer, device, **kwargs):
    inputs = tokenizer(texts, max_length=512, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    return [tokenizer.decode(hyp, skip_special_tokens=True) for hyp in hypotheses]


def qg_gen_batch(texts, model, tokenizer, device):
    return generate_batch([f"ask | {text}" for text in texts], model, tokenizer, device, max_length=32)


@torch.no_grad()
def qg_process_dataset(input_path, output_path, batch_size):
    model, tokenizer, device = qd_load_model()
    updated_samples = []

    with open(input_path, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    batch_texts = []
    batch_samples = []

    with tqdm(total=total_lines, desc="QG") as pbar:
        for line in lines:
            sample = json.loads(line)

            if sample.get("qg") is not None:
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
                    batch_questions = qg_gen_batch(batch_texts, model, tokenizer, device)
                    for i, question in enumerate(batch_questions):
                        batch_samples[i]["qg"] = question
                        updated_samples.append(batch_samples[i])
                except Exception as e:
                    print(f"Error processing batch: {e}")

                batch_texts, batch_samples = [], []

            pbar.update(1)

        if batch_texts:
            try:
                batch_questions = qg_gen_batch(batch_texts, model, tokenizer, device)
                for i, question in enumerate(batch_questions):
                    batch_samples[i]["qg"] = question
                    updated_samples.append(batch_samples[i])
            except Exception as e:
                print(f"Error processing final batch: {e}")

            finally:
                torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"QA complete. Processed {len(updated_samples)} samples.")


# if __name__ == "__main__":
#     input_file = "sample_data.jsonl"
#     output_file = "sample_data_qg.jsonl"

#     qg_process_dataset(input_file, output_file, batch_size=16)