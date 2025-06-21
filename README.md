# 🎤 TTS API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.112.1-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?style=for-the-badge&logo=pytorch)
![TTS](https://img.shields.io/badge/Coqui_TTS-0.22.0-FF6B6B?style=for-the-badge&logo=robot)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)

</div>

FastAPI tabanlı **Text-to-Speech** API servisi. Coqui AI'nin XTTS v2 modelini kullanarak yüksek kaliteli ses sentezleme sunar.

## 🚀 Hızlı Başlangıç

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Servisi başlat
python api.py
```

## 📡 API Endpoints

| Endpoint | Açıklama | Çıktı |
|----------|----------|--------|
| `POST /synthesize_file` | Metni WAV dosyası olarak döndürür | 🎵 Ses dosyası |
| `POST /synthesize_json` | Ses verisini JSON tensor olarak döndürür | 📊 JSON |
| `GET /stream_audio` | Canlı ses akışı | 📻 Stream |

## 💻 Kullanım Örneği

```bash
curl -X POST "http://127.0.0.1:8000/synthesize_file" \
  -H "Content-Type: application/json" \
  -d '{"text": "Merhaba dünya!", "speaker": "Dionisio Schuyler"}' \
  --output ses.wav
```

## 🛠️ Teknolojiler

- **FastAPI** - Modern web API framework
- **XTTS v2** - Çok dilli TTS modeli  
- **PyTorch** - ML backend

---
*Kullanıma hazır! Server: `http://localhost:8000`*
