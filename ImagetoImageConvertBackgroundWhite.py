from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Model ve dönüşümler
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

# verilen image işler 
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cpu")

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

# Resmi işleyip doğrudan gösteriyoruz.
if __name__ == "__main__":
    input_image_path = 'input.jpg'  # Giriş resminin yolu
    image = Image.open(input_image_path).convert('RGB')
    processed_image = process(image)
    
    # Sonuçları gösteren yer
    plt.imshow(processed_image)
    plt.axis('off')
    plt.show()
