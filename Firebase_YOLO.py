import time
import cv2
import os


from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore, storage



#veri yolu için zamana göre isimlendirme
veri_yol = ("/"+str(time.strftime('%c'))).replace(" ","-")


#veri yolu ve görüntü kaydetme sırası için yardımcı bir class, sadece veri tutacak
class Veri:
    yol = ""
    counter = 0

#Bir klasör yok ise oluşturma işlemi
if (not os.path.exists(os.getcwd() + veri_yol)):
    os.mkdir(os.getcwd() + veri_yol)

#yolo modeli yükle
model = YOLO("yolov8n.pt")


#firebase yapılandır
credentialData = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(credentialData, {
'storageBucket': 'firebase veya senin alannın.appspot.com'
})

#bilgileri buluta yükle
def firebase_veri_yukle(gorsel_Adi,nesne_sayi,tumBilgi):

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
            'gorsel_adi': "urun_Gorseller"+f"{gorsel_Adi}",
            'konum':"Yozgat/Boğazlıyan",
            'urun_kategori':nesne_sayi,
            'zaman': timestamp,
            "tumBilgi":tumBilgi,#YOLO bilgileri
        }
    
    #bilgi_upload
    belge_ref.set(urun_verileri, merge=True)

    #gorsel yükleme
    blob = bucket.blob("sune_gorseller"+gorsel_Adi)
    blob.upload_from_filename(gorsel_Adi)



    


#tahmin için gerekli fonskiyon
def goruntu_Tahmin(goruntu_yol):
    results = model([goruntu_yol])
    # Process results list
    for result in results:
        boxes = result.boxes
        print("\n\n\nBAŞLADI")
        print("HEPSİ BOXES")
        print(str(result.boxes))
        print("TUM BİLGİ")
        print(str(result))
        print("TİP")
        print(type(result.boxes))
        print("CLASS BİLGİSİ")
        print(result.boxes.cls)
        print("Toplam süne sayısı : "+str(len(result.boxes.cls)))
        print("CLASS ID BİLGİSİ")
        print(result.boxes.id)
        print("BİTTİ\n\n\n")
        result.save(filename=goruntu_yol)

        firebase_veri_yukle(goruntu_yol,str(len(result.boxes.cls)),str(result.boxes))


#Sürekli çalıştırma alanı
while True:



    #tahmin yapma
    goruntu_Tahmin(os.getcwd() + veri_yol + "/" +"image{}.jpg".format(Veri.counter))


    #yolları yazdır
    print("Veri yol")
    print(veri_yol)

    veri_yol_dosya = os.getcwd() + veri_yol + "/" + "image{}.jpg".format(Veri.counter)
    print("veri yol dosya")
    print(veri_yol_dosya)

    """
    Veri.yol = veri_yol_dosya
    cv2.imwrite(veri_yol_dosya,frame)
    """

    Veri.counter += 1


