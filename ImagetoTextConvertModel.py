from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch

model = AutoModelForCausalLM.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)

image = Image.open("input.jpg").convert("RGB")

prompt_list = [
  'Kısaca açıkla',
  'Detaylı açıkla',
  'Resimde ne görünüyor?',
  'Resimde ilgi çekici unsurlar nelerdir?',
]

inputs = processor(text=prompt_list, images=len(prompt_list)*[image], padding="longest", return_tensors="pt")

# Do not specify device_map or .to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2)

output_text_list = processor.batch_decode(outputs, skip_special_tokens=True)

for output_text in output_text_list:
  print(f"Model response: {output_text}\n\n\n")