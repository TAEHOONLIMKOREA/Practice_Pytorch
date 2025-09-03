import torch
from momentfm import MOMENTPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "reconstruction"},
)
model.to(device)
model.init()