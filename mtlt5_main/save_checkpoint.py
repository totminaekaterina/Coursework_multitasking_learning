import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR  = "/userspace/tev/cache/local-fredt5-model"
CKPT_PATH = "/userspace/tev/cache/checkpoints/epoch_1bln_2.pth"
SAVE_DIR  = "/userspace/tev/cache/checkpoints/model_1bln_2"

model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_DIR,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    low_cpu_mem_usage=True,
)

ckpt = torch.load(CKPT_PATH, map_location="cpu")

state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"missing: {len(missing)}, unexpected: {len(unexpected)}")

try:
    model.to("cuda", dtype=torch.bfloat16)
except RuntimeError:
    print("bf16 не поддерживается на вашей GPU – переключаюсь на float16")
    model.to("cuda", dtype=torch.float16)

model.save_pretrained(SAVE_DIR)
AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True).save_pretrained(SAVE_DIR)

print(f"Модель сохранена в {SAVE_DIR}")
