import json
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def mc_load_model():
    device = 0 if torch.cuda.is_available() else -1
    cache_dir = '/userspace/tev/cache/'

    model = AutoModelForSequenceClassification.from_pretrained(
        "classla/multilingual-IPTC-news-topic-classifier",
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "classla/multilingual-IPTC-news-topic-classifier",
        cache_dir=cache_dir
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=512,
        truncation=True
    )

    return classifier



@torch.no_grad()
def mc_predict_batch(texts, classifier):
    results = classifier(texts, truncation=True)
    if not isinstance(results, list):
        raise ValueError(f"Unexpected output format from classifier: {results}")
    return [result['label'] for result in results]


@torch.no_grad()
def mc_process_dataset(input_path, output_path, batch_size):
    classifier = mc_load_model()
    updated_samples = []

    with open(input_path, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    batch_texts = []
    batch_samples = []

    with tqdm(total=total_lines, desc="MC") as pbar:
        for line in lines:
            sample = json.loads(line)

            if sample.get("mc") is not None:
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
                    batch_classes = mc_predict_batch(batch_texts, classifier)
                    for i, mc_class in enumerate(batch_classes):
                        batch_samples[i]["mc"] = mc_class
                        updated_samples.append(batch_samples[i])
                except Exception as e:
                    print(f"Error processing batch: {e}")

                batch_texts, batch_samples = [], []

            pbar.update(1)

        if batch_texts:
            try:
                batch_classes = mc_predict_batch(batch_texts, classifier)
                for i, mc_class in enumerate(batch_classes):
                    batch_samples[i]["mc"] = mc_class
                    updated_samples.append(batch_samples[i])
            except Exception as e:
                print(f"Error processing final batch: {e}")

            finally:
                torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"MC complete. Processed {len(updated_samples)} samples.")


# if __name__ == "__main__":
#     input_file = "sample_data.jsonl"
#     output_file = "sample_data_mc.jsonl"

#     mc_process_dataset(input_file, output_file, batch_size=16)