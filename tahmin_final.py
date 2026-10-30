import cv2
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torchvision import models, transforms
from PIL import Image

# pip ile yüklemediğim için dosya yollarını koda tanıtmam gerek
sys.path.append('/home/ailoss/Depth-Anything-V2')
sys.path.append('/home/ailoss/segment-anything-2')
sys.path.append('/home/ailoss/segment-anything-2/sam2')

# Modelleri ve etiketleri import et
from turk_yemekleri_labels import TURKISH_LABELS
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from depth_anything_v2.dpt import DepthAnythingV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model mimarisi
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.base_model = models.resnet50(weights=None)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)



#Para tespiti
coin_model = YOLO('best.pt')

# SAM 2                   model mimarisi      hazır eğitilmiş model, ağırlık
sam2_model = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device=device)
predictor = SAM2ImagePredictor(sam2_model) #bir nevi hafıza hazır segmente edilen yeri değişkende tutmamıza yarıyor


# Depth Anything V2 - Derinlik tahmini (Large Model - Açık Parametreler)
depth_model = DepthAnythingV2(
    encoder='vitl', #model büyüklüğü
    features=256, #özellik çıkarmak için kullandığı kanal sayısı small modelde 64tü
    out_channels=[256, 512, 1024, 1024] #decoder bölümü 1024 çıkış keskin görüntü
    #encoder resmi küçültme maxpool buna örnek
)

depth_model.load_state_dict(torch.load('depth_anything_v2_vitl.pth', map_location='cpu'))#indirdiğimiz dosyayı yüklediğimiz kısım
depth_model.to(device).eval()

# Türk Yemekleri ResNet-50 
num_classes = len(TURKISH_LABELS) # toplam sınıf sayımız,

,
food_classifier = ResNetModel(num_classes).to(device) 
food_classifier.load_state_dict(torch.load('turk_yemekleri_resnet50.pth', map_location=device))
food_classifier.eval() #eval modeli kilitleyip train modundan tahmin moduna kilitler

# Kalori Veritabanı 
kcal_db = {"pizza": 2.66, "doner": 2.20, "baklava": 4.50, "default": 1.40}

def process_image(img_path):
    if not os.path.exists(img_path):
        return print(f"Hata: {img_path} bulunamadı!")

    raw_img = cv2.imread(img_path)#resimi okuyoruz
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)#open cv resimleri varsıylan bgr ama rgb üzerinden çalıştığımız için dönüşüm
    h, w = raw_img.shape[:2] # resmin çözünürlüğünü ayırır yükseklik ve genişlik olarak 1080x1920 gibi
    
    # PARA TESPİTİ 
    res = coin_model(raw_img)[0] #resmi yolonun içine atıp parayı buluyoruz
    if len(res.boxes) == 0: return print("Hata: 1 TL referans bulunamadı!") #eğer dönen değer sıfırsa para yoktur
    #para, .xyxy= sol üst ve sağ alt xy kordinatlarını diziye atar, en son değerler int çevrilir 
    #c_box[0] (x_min): Paranın sol kenarının X eksenindeki konumu.
    #c_box[1] (y_min): Paranın üst kenarının Y eksenindeki konumu.
    #c_box[2] (x_max): Paranın sağ kenarının X eksenindeki konumu.
    #c_box[3] (y_max): Paranın alt kenarının Y eksenindeki konumu.
    c_box = res.boxes[0].xyxy[0].cpu().numpy().astype(int)
    #26.15 gerçek hayattaki boyut çapı biz de xmax ve xmin i çıkararak görsel çapını bulup oranlıyoruz
    px_to_mm = 26.15 / (c_box[2] - c_box[0])
    
    # Görselleştirme
    yolo_viz = raw_img.copy()
    cv2.rectangle(yolo_viz, (c_box[0], c_box[1]), (c_box[2], c_box[3]), (0, 255, 0), 3)
    cv2.imwrite("cikti_1_yolo.jpg", yolo_viz)
    
    #YEMEK SEGMENTASYONU
    input_box = np.array([w*0.05, h*0.05, w*0.95, h*0.95]) #burda sam2 ye yardımcı olup odak noktası veriyoruz
    predictor.set_image(img_rgb)#resmi modelin içine yükler arkada resmi "embedding" matrise dönüştürür
    #maske, scores, logits(arka plandaki sayılar) sadece maske ile ilgilendiğimiz için gerisi önemli değil
    masks, _, _ = predictor.predict(box=np.array([input_box]), multimask_output=False)#birden fazla mask engelliyoruz
    
    mask = masks[0].astype(bool) #maske yemeğe ait mi değil mi kontrol 
    
    # Görselleştirme
    seg_viz = raw_img.copy()
    seg_viz[~mask] = seg_viz[~mask] // 4 #maskenin diğer taraflarını 4 e böl karanlık yap 
    cv2.imwrite("cikti_2_seg.jpg", seg_viz)
    
    #GIDA SINIFLANDIRMA
    y, x = np.where(mask) #segmente etiğimiz resmin x ve y koordinatlarını atıyoruz
    crop = img_rgb[np.min(y):np.max(y), np.min(x):np.max(x)] #kırpma işlemi yaparak cismi masadan ayırıyoruz
    tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#imagenet
    ])#resnet için uygun transpoz ediyoruz
    input_tensor = tr(crop).unsqueeze(0).to(device) #Yığın Sayısı, Kanal Sayısı, Yükseklik, Genişlik
    #yığın sayısına 1 atıyoruz unsqueeze(0) o yüzden kullanılıyor resnet gibi mimariler her zaman 4 boyutlu girdi ister
    with torch.no_grad():
        out = food_classifier(input_tensor)
        food_index = torch.argmax(out).item()#tahmin skoru en yüksek yemeğin indexini tutar
        food_name = TURKISH_LABELS[food_index]
    
    #DERİNLİK VE HACİM
    #derinlik haritası çıkarılır
    depth = model.infer_image(raw_img)#her bir pikselin kameraya olan uzaklığını tahmin eder inferin anlamı budur
    depth_n = (depth - depth.min()) / (depth.max() - depth.min())
    #depth min en uzak nokta
    #depth max en yakın nokta
    #burada yaptığımız işlem yukardan gelen sayıları 0 ve 1 lerin arasına sıkıştırmak= min max normalizasyonu
    #bunu yapmamızdaki sebep yukardan gelen sayılar bir resimde -200 ve 500 arasında olabilirken diğer resimde 1000 ve 8000 arasında olabilir bu yüzden evrensel bir skala oluşturuyoruz. bir nevi yüzdeliyoruz.
    
    # Görselleştirme
    depth_viz = cv2.applyColorMap((depth_n * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite("cikti_3_depth.jpg", depth_viz)
    
    #zemin seviyesini paranın merkezi olarak aylaradık depth_n[Y_merkez, X_merkez]
    ground_level = depth_n[int((c_box[1]+c_box[3])/2), int((c_box[0]+c_box[2])/2)]
    # 150.0 çarpanı derinlik birimini mm'ye çevirmek için kullanılan kalibrasyon değeridir
    height_mm = np.abs(depth_n[mask] - ground_level) * 150.0 #yükseklik için zeminle maskelediğimiz yeri çıkarıyoruz 150. abs mutlak değer alıyor
    vol_cm3 = np.sum(height_mm * (px_to_mm**2)) / 1000 #Riemann Toplamı 
    #px_to_mm**2 bir pikselin gerçek hayattaki hacimi
    #taban alanı * yükseklik en son mm^3 ü cm^3 e çeviriyoru
    
    # --- SONUÇLARIN YAZDIRILMASI ---
    kcal_per_cm3 = kcal_db.get(food_name, kcal_db["default"])#.get(aranan_kelime, yedek_plan)
    total_kcal = vol_cm3 * kcal_per_cm3
    
    print("\n" + "X"*40)
    print(f"YEMEK ANALİZ RAPORU")
    print(f"Tespit Edilen: {food_name.upper()}")
    print(f"Hacim: {vol_cm3:.2f} cm3")
    print(f"Tahmini Kalori: {total_kcal:.2f} kcal")
    print("X"*40)
    print("Analiz görselleri klasöre kaydedildi.")

if __name__ == "__main__":
    process_image("deneme.jpg")
