import json
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import torch
import warnings

warnings.filterwarnings("ignore")


def pa_load_model():
    MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
    cache_dir='/userspace/tev/cache/'
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir, legacy=False)
    model.cuda()
    model.eval()

    MAX_TOKENS = 400

    return model, tokenizer, MAX_TOKENS


def clean_generated_text(text):
    text = re.sub(r"(\\n)+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@torch.no_grad()
def paraphrase_batch(texts, model, tokenizer, beams=7, grams=4, do_sample=True, top_k=50, top_p=0.9, temperature=0.9, repetition_penalty=2.5):
    MAX_TOKENS = 400

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        encoder_no_repeat_ngram_size=grams,
        num_beams=beams,
        max_length=MAX_TOKENS,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )

    paraphrased_texts = [clean_generated_text(tokenizer.decode(out, skip_special_tokens=True)) for out in outputs]
    return paraphrased_texts


def pa_process_dataset(input_path, output_path, batch_size):
    model, tokenizer, _ = pa_load_model()
    updated_samples = []

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    for i in tqdm(range(0, len(lines), batch_size), desc="PA"):
        batch = lines[i:i + batch_size]
        texts = []
        indices_to_update = []
        current_batch_samples = []

        for index, line in enumerate(batch):
            sample = json.loads(line)
            current_batch_samples.append(sample)
            original_text = sample.get("text", "").strip()

            if not sample.get("paraphrase"):
                if original_text:
                    texts.append(original_text)
                    indices_to_update.append(index)

        if texts:
            try:
                paraphrased_texts = paraphrase_batch(texts, model, tokenizer)
                for idx, paraphrase in zip(indices_to_update, paraphrased_texts):
                    current_batch_samples[idx]["paraphrase"] = paraphrase
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
            finally:
                torch.cuda.empty_cache()

        updated_samples.extend(current_batch_samples)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"PA complete. Processed {len(updated_samples)} samples.")
    return True if updated_samples else False





# if __name__ == "__main__":
#     input_file = "sample_data.jsonl"
#     output_file = "sample_data_pa.jsonl"

#     pa_process_dataset(input_file, output_file, batch_size=8)