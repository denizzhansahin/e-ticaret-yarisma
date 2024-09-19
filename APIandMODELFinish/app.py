from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch
import PIL
import base64
from io import BytesIO
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from flask import Flask, request


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


model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
model = AutoModelForCausalLM.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)


app = Flask(__name__)

# API post method endpoint
@app.route('/UploadImage', methods=['POST'])
def upload_image_api():

    data = request.json
    base64_image = data.get('image') # body image

    # example send request 
    # { 
    # "image": "base64 format string"
    # }

    image = base64_to_image(base64_str=base64_image) # convert image file

    processed_image, output_text_lists = upload_image(image= image) # process MODELs

    base64_processed_image = image_to_base64(processed_image) # convert base64 processed_image

    return { # Response message API
        "image": base64_processed_image,
        "detected_text": [f"{text}, " 
            for text in output_text_lists
            ],
        "message": "Success",
        "code": 200
    }



# main function
def upload_image(image):
    
    original_image = image.convert('RGB')

    # Process the image and read image
    processed_image = process_image(original_image)
    output_text_lists = read_photo(image=processed_image)

    return processed_image, output_text_lists


# background remove and clear MODEL
def process_image(image):
        
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    
    pred = preds[0].squeeze() 
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)

    return image


# image detection MODEL
def read_photo(image):
    prompt_list = [
        'Ürünü kısaca açıkla',
        'Ürünü uzun açıkla'
    ]

    # Ensure the image is in the correct format (PIL Image)
    if not isinstance(image, PIL.Image.Image):
        raise ValueError("The image is not in the correct PIL format.")

    # Convert the image to a format compatible with the processor
    image = image.convert("RGB")

    # Prepare inputs for the model
    inputs = processor(text=prompt_list, images=[image] * len(prompt_list), padding="longest", return_tensors="pt")

    # Generate outputs
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2)

    output_text_list = processor.batch_decode(outputs, skip_special_tokens=True)

    return output_text_list

# convert base64 to image file
def base64_to_image(base64_str):
    if base64_str.startswith('data:image/jpeg;base64,'):
        base64_str = base64_str.replace('data:image/jpeg;base64,', '')
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image


# convert image -> PNG format to base64 
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# main Flask app run
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=1054)