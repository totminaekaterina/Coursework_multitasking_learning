import torch
from transformers import AutoModelForSeq2SeqLM

# 1) инициализируем базовую архитектуру
model = AutoModelForSeq2SeqLM.from_pretrained("/userspace/tev/cache/local-fredt5-model")

# 2) загружаем свои веса
state = torch.load(r"/userspace/tev/cache/output/epoch_4.pth", map_location="cuda", weights_only=True)
model.load_state_dict(state, strict=False)

# 3) сохраняем в HF‑папку
model.save_pretrained(r"/userspace/tev/cache/output/model_10k/")