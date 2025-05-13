import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import tqdm


def title_load_model():
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


def generate_title(texts, model, tokenizer, device):
    return generate_batch([f"headline | {text}" for text in texts], model, tokenizer, device, max_length=32)


def title_process_dataset(input_file, output_file, batch_size=8):
    model, tokenizer, device = title_load_model()
    updated_samples = []

    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    for i in tqdm(range(0, len(lines), batch_size), desc="TITLE"):
        batch = lines[i:i + batch_size]
        texts = []
        samples = []

        for line in batch:
            sample = json.loads(line)
            original_text = sample.get("text", "").strip()

            if sample.get("title") is None:
                if original_text:
                    texts.append(original_text)
                    samples.append(sample)
            else:
                updated_samples.append(sample)

        if texts:
            try:
                generated_titles = generate_title(texts, model, tokenizer, device)
                for sample, title in zip(samples, generated_titles):
                    sample["title"] = title
                    updated_samples.append(sample)
            except Exception as e:
                print(f"Ошибка генерации заголовков для батча: {e}")
            finally:
                torch.cuda.empty_cache()

    if updated_samples:
        try:
            with open(output_file, "w", encoding="utf-8") as outfile:
                for sample in updated_samples:
                    outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"Processing complete. Processed {len(updated_samples)} samples.")
        except Exception as e:
            print(f"Ошибка записи в файл {output_file}: {e}")
    else:
        print("Не было обработано ни одной записи.")


# if __name__ == "__main__":
#     input_file = "sample_data.jsonl"
#     output_file = "sample_data_title.jsonl"

#     title_process_dataset(input_file, output_file, batch_size=8)