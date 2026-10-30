# %% kütüphaneler 
import torch # tensor(matris) işlemleri
import torch.nn as nn # yapay sinir agi katmalarını tanılıyoruz
import torch.optim as optim  # optimizasyon algoritmalarını içeren modül
import torchvision # görüntü işleme ve önceden eğitilmiş modelleri içerir
import torchvision.transforms as transforms # görüntü donusumleri yapmak
from torchvision import models # resnet modeli için gerekli
from torchvision.datasets import ImageFolder #kendi yüklediğim verisetini kullanmak için
from torch.utils.data import DataLoader, random_split # Veri bölmek için eklendi
import matplotlib.pyplot as plt #görselleştirme
import numpy as np 
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışılan Cihaz: {device}")

# %% veri seti yükleme 

def get_data_loaders(batch_size = 32): 
 
    mean_vals = (0.485, 0.456, 0.406)
    std_vals = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),   
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(15),           
        transforms.ToTensor(), 
        transforms.Normalize(mean_vals, std_vals) 
    ])

    # Yeni veri seti yolumuz
    data_dir = os.path.expanduser("~/Masaüstü/Tasarım 1 Turkyemekleri/Turkish Food")

    # klasör kontrolü
    if not os.path.exists(data_dir):
        print("UYARI: Klasör bulunamadı! Lütfen dosya yolunu kontrol et.")
        return None, None, 0

    # veri setini yükleme (Tüm veri)
    full_dataset = ImageFolder(root=data_dir, transform=train_transform)

    # İleride kullanmak için sınıf isimlerini bir python dosyasına kaydediyoruz
    class_names = full_dataset.classes
    with open('turk_yemekleri_labels.py', 'w') as f:
        f.write(f"TURKISH_LABELS = {class_names}")
    print(f"Toplam {len(class_names)} sınıf bulundu ve 'turk_yemekleri_labels.py' dosyasına kaydedildi.")

    # Veriyi %80 Eğitim, %20 Test olarak ayırıyoruz
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(full_dataset, [train_size, test_size])

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(class_names)

# %% Model Mimarisi (ResNet-50)

class ResNetModel(nn.Module):

    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        
        # Önceden eğitilmiş ResNet-50 modelini indiriyoruz
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Son katmanı kendi sınıfımıza göre (104) değiştiriyoruz
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes) 

    def forward(self, x):
        x = self.base_model(x)
        return x

# %% loss fonksiyonu oluşturma

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), 
    optim.Adam(model.parameters(), lr = 0.001) 
)

# %% modelin eğitilmesi / training

def train_model(model, train_loader, criterion, optimizer, epochs = 50): 

    model.train() 
    train_losses = [] 

    for epoch in range(epochs): 
        total_loss = 0 
        
        print(f"Epoch {epoch+1}/{epochs} başlıyor...")

        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() 

            outputs = model(images) 

            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"--> Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

    # kayıp grafiğini çizdireceğiz
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("ResNet-50 Training Loss")
    plt.legend()
    plt.savefig('resnet50_result.png') 
    print("Grafik kaydedildi: resnet50_result.png")
    # plt.show() # Sunucuda hata vermemesi için show() komutunu kapattım, png olarak kaydedecek.
    
# %% modelin test edilmesi

def test_model(model, test_loader, dataset_type="test"):
    
    model.eval() 
    correct = 0 
    total = 0 
    
    with torch.no_grad(): 
        for images, labels in test_loader: 
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images) 
            _, predicted = torch.max(outputs, 1) 
            
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
            
    print(f"{dataset_type} accuracy: {100* correct / total:.2f}%")
            
# %% main 

if __name__ == "__main__":

    # veri seti yükleme
    train_loader, test_loader, num_classes = get_data_loaders(batch_size=32)
    
    if train_loader is not None:
        # training
        model = ResNetModel(num_classes).to(device) 
        
        criterion, optimizer = define_loss_and_optimizer(model)
        
        train_model(model, train_loader, criterion, optimizer, epochs=50) 

        # modeli kaydediyoruz
        torch.save(model.state_dict(), 'turk_yemekleri_resnet50.pth')
        print("Model kaydedildi: turk_yemekleri_resnet50.pth")

        # test
        test_model(model, test_loader, dataset_type = "Test Seti")
        test_model(model, train_loader, dataset_type = "Training Seti")
