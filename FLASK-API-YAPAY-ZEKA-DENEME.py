#KURULUMLAR
"""
pip install -U "huggingface_hub[cli]"


!huggingface-cli login    ::::: buraya hugging face şifreni gir


#kurulumlar
!pip install timm kornia diffusers


!pip install firebase_admin


!pip install ultralytics
"""

import time

from firebase_admin import credentials, firestore, storage, db

import cv2
import os
from ultralytics import YOLO

#firebase yapılandır
credentialData = credentials.Certificate("hackathonproject-5e6ef-firebase-adminsdk-mubj0-d18674376c.json")
firebase_admin.initialize_app(credentialData, {
'storageBucket': 'hackathonproject-5e6ef.appspot.com',
'databaseURL': 'https://hackathonproject-5e6ef-default-rtdb.firebaseio.com/'
}
)



#TUM VERİ CLASS
class Veri():
    orijinal_goruntu = None
    yolo_goruntu = None
    yolo_goruntu_bilgi = None
    arka_plan_ayrilmis = None
    iyilesitirilmis_goruntu = None
    goruntu_bilgisi = None
    firebase_veri_document_sayisi = None
    urunKategori = None


veri_yol = ("/"+str(time.strftime('%c'))).replace(" ","-")

#BASE64 işlemleri
import base64
from PIL import Image

def base64_to_image(base64_string, output_file=str(Veri.firebase_veri_document_sayisi)+'output.png'):
  img_data = base64.b64decode(base64_string)
  with open(output_file, 'wb') as f:
    f.write(img_data)
  img = Image.open(output_file)
  img.show()
  Veri.orijinal_goruntu = output_file

ref = db.reference('/')
import time


def kod(veri):
  print("merhaba")
  base64_to_image(veri)
  time.sleep(10)





#bilgileri buluta yükle
def firebase_veri_yukle():

    firestoreDb = firestore.client()
    bucket = storage.bucket()


    #document sayısını öğrenme
    veri_document_sayi = firestoreDb.collection("urunVeri").get()

    #firebase tarih bilgisi
    timestamp = firestore.SERVER_TIMESTAMP

    #veri girişi
    deger = len(veri_document_sayi)+1
    belge_ref = firestoreDb.collection("urunVeri").document("urunBilgi"+str(deger))


    #veriler
    urun_verileri = {
            'ID':str(deger),
            'orijinal_gorsel_adi': "urun_Gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.orijinal_goruntu,
            'arka_plan_ayrilmis_gorsel_adi':"urun_Gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.arka_plan_ayrilmis,
            'iyilesitirilmis_gorsel_adi':"urun_Gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.iyilesitirilmis_goruntu,
            'yolo_gorsel_adi':"urun_Gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.yolo_goruntu,
            'yolo_gorsel_bilgisi':Veri.yolo_goruntu_bilgi,

            'konum':"Yozgat/Boğazlıyan",
            'urun_kategori':Veri.urunKategori,
            'zaman': timestamp,
            "kisaBilgi":Veri.goruntu_bilgisi[0],
            "detayBilgi":Veri.goruntu_bilgisi[1],
            "resimBilgi":Veri.goruntu_bilgisi[2],
            "ilgiCekiciUnsurlar":Veri.goruntu_bilgisi[3],
        }

    #bilgi_upload
    belge_ref.set(urun_verileri, merge=True)

    #gorsel yükleme
    blob = bucket.blob("urun_gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.orijinal_goruntu)
    blob.upload_from_filename(Veri.orijinal_goruntu)

    blob = bucket.blob("urun_gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.yolo_goruntu)
    blob.upload_from_filename(Veri.yolo_goruntu)

    blob = bucket.blob("urun_gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.arka_plan_ayrilmis)
    blob.upload_from_filename(Veri.arka_plan_ayrilmis)

    blob = bucket.blob("urun_gorseller"+str(Veri.firebase_veri_document_sayisi)+veri_yol+Veri.iyilesitirilmis_goruntu)
    blob.upload_from_filename(Veri.iyilesitirilmis_goruntu)




names= {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


#tahmin için gerekli fonskiyon
def goruntu_Tahmin(goruntu_yol,model):
    results = model([goruntu_yol])
    # Process results list
    for result in results:
        boxes = result.boxes
        print("\n\n\nBAŞLADI")
        print("CLASS BİLGİSİ")
        print(result.boxes.cls)
        print(int(result.boxes.cls[0]))
        print(names[int(result.boxes.cls[0])])
        print("Toplam nesne sayısı : "+str(len(result.boxes.cls)))
        print("CLASS ID BİLGİSİ")
        print(result.boxes.id)
        print("BİTTİ\n\n\n")
        result.save(filename=str(Veri.firebase_veri_document_sayisi)+goruntu_yol)
        Veri.urunKategori = names[int(result.boxes.cls[0])]
        Veri.yolo_goruntu = str(Veri.firebase_veri_document_sayisi)+goruntu_yol
        Veri.yolo_goruntu_bilgi = str(result.boxes)
        Veri.orijinal_goruntu = goruntu_yol



#yolo modeli yükle
def YOLO_islem(gorsel_yolu):
    model = YOLO("yolov8n.pt")
    #tahmin yapma
    goruntu_Tahmin(gorsel_yolu,model)

    #yolları yazdır
    print("Veri yol")
    print(gorsel_yolu)




#ARKA PLAN AYIRMA
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision(["high", "highest"][0])
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")
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
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)



#GORUNTU BİLGİ ALMA
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch
model = AutoModelForCausalLM.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('ucsahin/TraVisionLM-DPO', trust_remote_code=True)


#GORUNTU ARKA PLAN AYIRMA
# verilen image işler
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

# Resmi işleyip doğrudan gösteriyoruz.
def arka_plan_ayirma():
    input_image_path = Veri.orijinal_goruntu  # Giriş resminin yolu
    image = Image.open(input_image_path).convert('RGB')
    processed_image = process(image)
    # Sonuçları gösteren yer
    plt.imshow(processed_image)
    plt.axis('off')
    Veri.arka_plan_ayrilmis = Veri.orijinal_goruntu + str(Veri.firebase_veri_document_sayisi)+'.png'  # Çıkış resminin yolu
    plt.savefig(Veri.arka_plan_ayrilmis)
    plt.show()



#GORUNTU IYILESTİRME
def goruntu_acma(deger):
    image = PIL.Image.open(deger)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def goruntu_iyilestirme():
  image = goruntu_acma(Veri.arka_plan_ayrilmis)
  prompt = "make the image as real as if it was taken from a color camera"
  images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
  images[0]
  Veri.iyilesitirilmis_goruntu = Veri.arka_plan_ayrilmis +str(Veri.firebase_veri_document_sayisi)+'.png'
  images[0].save(Veri.iyilesitirilmis_goruntu)


#GORUNTU BİLGİ ALMA
def goruntu_bilgisi_alma():
  image = Image.open(Veri.arka_plan_ayrilmis).convert("RGB")
  print("BİLGİ ALMA")

  prompt_list = [
    'Kısaca açıkla',
    'Detaylı açıkla',
    'Resimde ne görünüyor?',
    'Resimde ilgi çekici unsurlar nelerdir?',
  ]

  print("BİLGİ ALMA 1")
  inputs = processor(text=prompt_list, images=len(prompt_list)*[image], padding="longest", return_tensors="pt")

  print("BİLGİ ALMA 2")
  # Do not specify device_map or .to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2)

  print("BİLGİ ALMA 3")
  output_text_list = processor.batch_decode(outputs, skip_special_tokens=True)

  print("BİLGİ ALMA 4")
  for output_text in output_text_list:
    print(f"Model response: {output_text}\n\n\n")

  print("BİLGİ ALMA 5")
  Veri.goruntu_bilgisi = output_text_list
  print("BİLGİ ALMA 6")



def ana_kod(veri):
    #ANA KOD
    firestoreDb = firestore.client()
    bucket = storage.bucket()

    #document sayısını öğrenme
    veri_document_sayi = firestoreDb.collection("urunVeri").get()

    #veri girişi
    deger = len(veri_document_sayi)+1
    belge_ref = firestoreDb.collection("urunVeri").document("urunBilgi"+str(deger))

    Veri.firebase_veri_document_sayisi = deger

    time.sleep(10)

    Veri.firebase_veri_document_sayisi = deger
    kod(veri)

    #ref.update({"message": "yeni_değer"})
    #ref.update({"base64": "yeni_değer"})
    time.sleep(10)

    #base64 olarak kayıt edilen veriyi çalıştır
    YOLO_islem(Veri.orijinal_goruntu)
    arka_plan_ayirma()
    goruntu_iyilestirme()
    goruntu_bilgisi_alma()
    print(Veri.orijinal_goruntu)
    print(Veri.yolo_goruntu)
    print(Veri.yolo_goruntu_bilgi)
    print(Veri.arka_plan_ayrilmis)
    print(Veri.iyilesitirilmis_goruntu)
    print(Veri.goruntu_bilgisi)
    print(Veri.firebase_veri_document_sayisi)
    print(Veri.urunKategori)
    firebase_veri_yukle()



from flask import Flask, request

# Flask app creation
app = Flask(__name__)

# Routes for query examples
@app.route('/')
def anasayfa():
    request  # This line might be unnecessary (check functionality)
    return 'Merhaba nasılsın'

@app.route('/komut')
def komut():
    deger = request.args.get('deger')
    if(deger!=None):
        ana_kod(deger)
        return 'merhaba... {}'.format(deger)  # Remove indentation here
    else:
        return 'merhaba... {}'.format(deger)


if __name__ == "__main__":
    app.run(debug=True)