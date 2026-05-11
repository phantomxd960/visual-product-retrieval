import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-flan-t5-xl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Downloading processor...")
processor = Blip2Processor.from_pretrained(model_name)

print("Downloading BLIP-2 model (this may take ~15 minutes)...")

model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

print("BLIP-2 downloaded successfully!")