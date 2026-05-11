import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-flan-t5-xl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load processor
processor = Blip2Processor.from_pretrained(model_name, use_fast=True)

# load model
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.float16
)

model.to(device)
model.eval()

print("Model loaded on:", next(model.parameters()).device)