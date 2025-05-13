import json
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def qa_load_model():
    device = 0 if torch.cuda.is_available() else -1
    cache_dir = '/userspace/tev/cache/'

    model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2", cache_dir=cache_dir)

    qa_model = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return qa_model



def qa_gen_batch(questions, contexts, qa_model):
    results = []
    for question, context in zip(questions, contexts):
        try:
            answer = qa_model(question=question, context=context)
            results.append(answer["answer"])
        except Exception as e:
            print(f"Error processing question-context pair: {question}, {context} - {e}")
            results.append(None)
    return results


@torch.no_grad()
def qa_process_dataset(input_path, output_path, batch_size):
    qa_model = qa_load_model()
    updated_samples = []

    with open(input_path, "r", encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    batch_questions = []
    batch_contexts = []
    batch_samples = []

    with tqdm(total=total_lines, desc="QA") as pbar:
        for line in lines:
            sample = json.loads(line)

            if sample.get("qa") is not None:
                updated_samples.append(sample)
                pbar.update(1)
                continue

            context = sample.get("text", "").strip()
            question = sample.get("qg", "").strip()
            if not context or not question:
                print(f"Skipping invalid sample: {sample}")
                pbar.update(1)
                continue

            batch_questions.append(question)
            batch_contexts.append(context)
            batch_samples.append(sample)

            if len(batch_questions) == batch_size:
                batch_answers = qa_gen_batch(batch_questions, batch_contexts, qa_model)
                for i, answer in enumerate(batch_answers):
                    batch_samples[i]["qa"] = answer
                    updated_samples.append(batch_samples[i])

                batch_questions, batch_contexts, batch_samples = [], [], []

            pbar.update(1)

        if batch_questions:
            batch_answers = qa_gen_batch(batch_questions, batch_contexts, qa_model)
            for i, answer in enumerate(batch_answers):
                batch_samples[i]["qa"] = answer
                updated_samples.append(batch_samples[i])

            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processing complete. Processed {len(updated_samples)} samples.")


# if __name__ == "__main__":
#     input_file = "sample_data_qg.jsonl"
#     output_file = "sample_data_qa.jsonl"

#     qa_process_dataset(input_file, output_file, batch_size=16)