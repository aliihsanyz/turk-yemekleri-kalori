from flask import Flask, request, jsonify

app = Flask(__name__)

# Flutter'ın istek yapacağı adres
@app.route('/analiz', methods=['POST'])
def analiz_et():
    # 1. Gelen isteğin içinde 'resim' adında bir dosya var mı kontrol et
    if 'resim' not in request.files:
        return jsonify({"hata": "Resim gönderilmedi!"}), 400
    
    dosya = request.files['resim']
    
    if dosya.filename == '':
        return jsonify({"hata": "Geçersiz dosya!"}), 400

    # TODO: Gerçek yapay zeka (ResNet ve MiDaS) kodlarını 
    # ve integral hesaplamalarını buraya ekleyeceğiz.
    # Şimdilik Flutter'ın bizimle konuşabildiğini test etmek için 
    # sahte (dummy) bir sonuç döndürüyoruz.
    
    hesaplanan_sonuc = {
        "yemek_adi": "TEST PİZZA",
        "hacim_cm3": 345.6,
        "kalori": 920.5,
        "mesaj": "Kalori hesabı başarıyla tamamlandı!"
    }
    
    # Sonucu JSON (Mobilin anlayacağı dil) formatında geri gönder
    return jsonify(hesaplanan_sonuc)

if __name__ == '__main__':
    # Sunucuyu yerel ağda dinlemeye başla
    print("Yapay Zeka Sunucusu Başlatılıyor...")
    app.run(host='0.0.0.0', port=5000, debug=True)
