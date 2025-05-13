import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def nli_load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    cache_dir='/userspace/tev/cache/'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir, legacy=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, cache_dir=cache_dir)

    if torch.cuda.is_available():
        model.to(device)

    return model, tokenizer, device


def calculate_nli_scores(batch_text1, batch_text2, model, tokenizer, device):
    with torch.inference_mode():
        encoded = tokenizer(batch_text1, batch_text2, padding=True, truncation=True, return_tensors='pt').to(device)
        out = model(**encoded)
        probas = torch.softmax(out.logits, -1).cpu().numpy()

        results = []
        for proba in probas:
            max_proba = proba.max()
            predicted_class = model.config.id2label[proba.argmax()]
            if max_proba < 0.7:
                results.append("neutral")
            else:
                results.append(predicted_class)
        return results


@torch.no_grad()
def nli_process_dataset(input_path, output_path, batch_size):
    model, tokenizer, device = nli_load_model()
    updated_samples = []

    with open(input_path, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    batch_text1 = []
    batch_text2 = []
    batch_samples = []

    with tqdm(total=total_lines, desc="NLI") as pbar:
        for line in lines:
            sample = json.loads(line)

            if sample.get("nli") is not None:
                updated_samples.append(sample)
                pbar.update(1)
                continue

            original_text = sample.get("text", "").strip()
            paraphrased_text = sample.get("paraphrase", "").strip()
            if not original_text or not paraphrased_text:
                pbar.update(1)
                continue

            batch_text1.append(original_text)
            batch_text2.append(paraphrased_text)
            batch_samples.append(sample)

            if len(batch_text1) == batch_size:
                batch_nli_labels = calculate_nli_scores(batch_text1, batch_text2, model, tokenizer, device)
                for i, label in enumerate(batch_nli_labels):
                    batch_samples[i]["nli"] = label
                    updated_samples.append(batch_samples[i])
                batch_text1, batch_text2, batch_samples = [], [], []

                torch.cuda.empty_cache()

            pbar.update(1)

        if batch_text1:
            batch_nli_labels = calculate_nli_scores(batch_text1, batch_text2, model, tokenizer, device)
            for i, label in enumerate(batch_nli_labels):
                batch_samples[i]["nli"] = label
                updated_samples.append(batch_samples[i])

            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"NLI complete. Processed {len(updated_samples)} samples.")


# if __name__ == "__main__":
#     input_file = "sample_data.jsonl"
#     output_file = "sample_data_nli.jsonl"

#     nli_process_dataset(input_file, output_file, batch_size=16)