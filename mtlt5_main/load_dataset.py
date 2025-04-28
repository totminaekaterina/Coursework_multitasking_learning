import json
from tqdm import tqdm


def process_multitask_dataset(input_path, batch_size):
    """
    Прочитать JSONL файл и объединить задачи для каждого примера данных.
    """
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()


    train_data = []


    for i in tqdm(range(0, len(lines), batch_size), desc="Processing"):
        batch = lines[i:i + batch_size]


        for line in batch:
            try:
                sample = json.loads(line.strip())  # Убедимся, что строка корректно десериализуется
                print("Sample keys:", sample.keys())
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue


            entry = {
                "text": sample.get("text", "").strip(),
                "title": sample.get("title", "").strip(),
                "paraphrase": sample.get("paraphrase", "").strip(),
                "sum": sample.get("sum", "").strip(),
                "qg": sample.get("qg", "").strip(),
                "qa": sample.get("qa", "").strip(),
                "nli": sample.get("nli", "").strip(),
                "mc": sample.get("mc", "").strip(),
                "sts": sample.get("sts", ),
                "ner": sample.get("ner", [])
            }


            # Удаляем пустые задачи
            entry = {k: v for k, v in entry.items() if v}


            # Проверяем, что метка NLI корректна
            if "nli" in entry:
                valid_nli_labels = {"contradiction", "entailment", "neutral"}
                if entry["nli"] not in valid_nli_labels:
                    print(f"Invalid NLI label '{entry['nli']}' in entry: {entry}")
                    break  # Пропускаем некорректные данные


            # Проверяем, что метка MC корректна
            if "mc" in entry:
                valid_mc_labels = {
                    'education', 'human interest', 'society', 'sport', 'crime, law and justice',
                    'disaster, accident and emergency incident', 'arts, culture, entertainment and media',
                    'politics', 'economy, business and finance', 'lifestyle and leisure',
                    'science and technology', 'health', 'labour', 'religion', 'weather',
                    'environment', 'conflict, war and peace'
                }
                if entry["mc"] not in valid_mc_labels:
                    print(f"Invalid MC label '{entry['mc']}' in entry: {entry}")
                    break


            # Проверяем, что метки NER корректны
            if "ner" in entry:
                valid_ner_tags = {
                    'AGE',
                    'AWARD',
                    'CITY',
                    'COUNTRY',
                    'CRIME',
                    'DATE',
                    'DISEASE',
                    'DISTRICT',
                    'EVENT',
                    'FACILITY',
                    'FAMILY',
                    'IDEOLOGY',
                    'LANGUAGE',
                    'LAW',
                    'LOCATION',
                    'MONEY',
                    'NATIONALITY',
                    'NUMBER',
                    'ORDINAL',
                    'ORGANIZATION',
                    'PENALTY',
                    'PERCENT',
                    'PERSON',
                    'PRODUCT',
                    'PROFESSION',
                    'RELIGION',
                    'STATE_OR_PROVINCE',
                    'TIME',
                    'WORK_OF_ART'}
                for entity in entry["ner"]:
                    if entity["tag"] not in valid_ner_tags:
                        print(f"Invalid NER tag '{entity['tag']}' in entry: {entry}")
                        entry["ner"].remove(entity)  # Удаляем некорректную сущность
                        continue  # Продолжаем обработку


            train_data.append(entry)
            # print(train_data[0])


    return train_data

