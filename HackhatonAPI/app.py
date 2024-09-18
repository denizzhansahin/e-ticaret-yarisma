#ARKA PLAN AYIRMA
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision(["high", "highest"][0])
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cpu")
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

#GORSEL ÖZELLİKLERİNİ DÜZENLEME
#https://huggingface.co/timbrooks/instruct-pix2pix
import PIL
from flask import Flask, json, request
import base64
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)



#GORUNTU BİLGİ ALMA
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import io
import torch
model = AutoModelForCausalLM.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)
     

class upload_image_response:
   def __init__(self, base64_image:str, output_text_lists: list[str]) -> None:
      self.base64_image = base64_image
      self.output_text_lists = output_text_lists


app = Flask(__name__)


@app.route('/UploadImage', methods=['POST'])
def upload_image():
    response = {}
    data = request.json
    image_base64 = data.get('image')

    try:
        # Decode the base64 image
        original_image_bytes = base64.b64decode(image_base64)
        # Convert bytes to a PIL Image
        original_image = Image.open(io.BytesIO(original_image_bytes))

        # Process the image
        processed_image, output_text_lists = process(original_image)

        # Convert the processed image back to bytes and then to base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")  # or whatever format you need
        return_base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        upload_image_model = upload_image_response(base64_image=return_base64_image, output_text_lists=output_text_lists)
        
        response = {
            "data": json.dumps(upload_image_model.__dict__),
            "message": "Success",
            "code": 200
        }

    except Exception as e:
        response = {
            "message": "An error occurred while processing the image.",
            "code": 500,
            "error": str(e)
        }

    return response



#GORUNTU ARKA PLAN AYIRMA
# verilen image işler
def process(image):
    try:
        # Convert the image to RGB if it has an alpha channel or is in another format
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')

        image_size = image.size
        input_images = transform_image(image).unsqueeze(0).to("cpu")

        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        
        print(f"Input shape: {input_images.shape}, Predictions shape: {preds.shape}")

        pred = preds[0].squeeze()  # Shape will be [1, 1024, 1024]
        
        # Convert the single-channel prediction to a binary mask
        binary_mask = (pred > 0.5).float()  # Thresholding to create a binary mask
        
        # Expand dimensions to create an RGB mask
        rgb_mask = binary_mask.repeat(3, 1, 1)  # Shape will be [3, 1024, 1024]
        
        # Convert to PIL image
        mask_pil = transforms.ToPILImage()(rgb_mask)
        mask = mask_pil.resize(image_size)
        
        # Put the mask as alpha (if you want to keep it transparent)
        image.putalpha(mask)

        # Image enhancement
        image = PIL.ImageOps.exif_transpose(image)
        image_text_lists = goruntu_bilgisi_alma(image=image)

        return image, image_text_lists
    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        raise




#GORUNTU BİLGİ ALMA
def goruntu_bilgisi_alma(image):

  prompt_list = [
     'Ürünü kısaca açıkla',
     'Ürünü uzun açıkla'
  ]

  inputs = processor(text=prompt_list, images=len(prompt_list)*[image], padding="longest", return_tensors="pt")

  # Do not specify device_map or .to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2)

  output_text_list = processor.batch_decode(outputs, skip_special_tokens=True)

  for output_text in output_text_list:
    print(f"Model response: {output_text}\n\n\n")

  return output_text_list  


if __name__ == '__main__':
   app.run(host='10.19.10.194', port=1090, debug=True)

