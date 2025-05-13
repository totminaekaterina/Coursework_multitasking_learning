import logging
import warnings
from GenData.MC_gen import mc_process_dataset
from GenData.NER_Nerel import Ner_Extractor, ner_process_dataset
from GenData.NLI_gen import nli_process_dataset
from GenData.PA_gen import pa_process_dataset
from GenData.QA_gen import qa_process_dataset
from GenData.QD_gen import qg_process_dataset
from GenData.SUM_gen import sum_process_dataset
from GenData.gen_title import title_process_dataset

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_files = [
    "train.jsonl",
    "val.jsonl"
]

for i, input_file in enumerate(input_files, start=5):
    logging.info(f"Обработка файла: {input_file}")
    
    temp_file = "sample_data_temp.jsonl"
    output_file = f"data_out_{i}.jsonl"
    
    pa_process_dataset(input_file, temp_file, batch_size=64)
    sum_process_dataset(temp_file, temp_file, batch_size=64)
    title_process_dataset(temp_file, temp_file, batch_size=64)
    ner_process_dataset(temp_file, temp_file, batch_size=64)
    nli_process_dataset(temp_file, temp_file, batch_size=64)
    qg_process_dataset(temp_file, temp_file, batch_size=64)
    qa_process_dataset(temp_file, temp_file, batch_size=64)
    mc_process_dataset(temp_file, output_file, batch_size=64)
    
    logging.info(f"Файл {input_file} обработан и сохранен в {output_file}")

logging.info("Все файлы обработаны.")
