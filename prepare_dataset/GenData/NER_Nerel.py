import json
from tqdm import tqdm
from transformers import pipeline
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

class Ner_Extractor:
    def __init__(self, model_checkpoint: str):
        cache_dir = '/userspace/tev/cache/'
        self.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir)

        self.token_pred_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="average",
            device=0 if torch.cuda.is_available() else -1
        )


    @staticmethod
    def concat_entities(ner_result, text):
        entities = []
        words = text.split()
        word_positions = []

        current_pos = 0
        for word in words:
            start = text.find(word, current_pos)
            end = start + len(word)
            word_positions.append((start, end))
            current_pos = end

        for ner in ner_result:
            word_index = -1
            for idx, (start, end) in enumerate(word_positions):
                if start <= ner["start"] < end:
                    word_index = idx
                    break

            word = text[ner["start"]:ner["end"]]
            entities.append({"word": word, "tag": ner["entity_group"], "index": word_index})

        return entities

    def get_entities(self, text: str):
        assert len(text) > 0, "Input text is empty."
        ner_result = self.token_pred_pipeline(text)
        return self.concat_entities(ner_result, text)


def ner_process_dataset(input_path, output_path, batch_size=16):
    extractor = Ner_Extractor(model_checkpoint="surdan/LaBSE_ner_nerel")
    updated_samples = []

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    for i in tqdm(range(0, len(lines), batch_size), desc="NER"):
        batch = lines[i:i + batch_size]
        texts = []
        indices_to_update = []
        current_batch_samples = []

        for index, line in enumerate(batch):
            sample = json.loads(line)
            current_batch_samples.append(sample)
            original_text = sample.get("text", "").strip()

            if sample.get("ner") is None:
                if original_text: 
                    texts.append(original_text)
                    indices_to_update.append(index)

        if texts:
            try:
                ner_results = [extractor.get_entities(text) for text in texts]
                for idx, ner_result in zip(indices_to_update, ner_results):
                    current_batch_samples[idx]["ner"] = ner_result
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
            finally:
                torch.cuda.empty_cache()

        updated_samples.extend(current_batch_samples)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"NER complete. Processed {len(updated_samples)} samples.")


# if __name__ == "__main__":
#     input_file = "sample_data.jsonl"
#     output_file = "sample_data_ner.jsonl"

#     ner_process_dataset(input_file, output_file, batch_size=16)