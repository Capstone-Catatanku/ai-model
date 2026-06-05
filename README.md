# AI Model - Catatanku

Repository ini berisi dua model AI yang dikembangkan untuk aplikasi **Catatanku**, sebuah aplikasi pencatatan keuangan pribadi. Kedua model bekerja secara terpisah dan saling melengkapi untuk memberikan pengalaman pengelolaan keuangan yang lebih cerdas.

---

## 📁 Struktur Repository

```
ai-model-main/
├── ai-model-tabungan/       # Model prediksi sisa cicilan tabungan
│   ├── model-ai-tabungan.ipynb
│   ├── model_lstm_tabungan.keras
│   ├── best_lstm_weights.weights.h5
│   ├── scaler_lstm.pkl
│   ├── scaler_y_lstm.pkl
│   ├── Data_Clean.csv
│   ├── logs/
│   └── requirements.txt
│
└── ai_klasifikasi/          # Model klasifikasi kategori transaksi
    ├── Catatanku_Klasifikasi.ipynb
    ├── model_fitur_klasifikasi.keras
    ├── label_encoder.pkl
    ├── vocabulary.json
    └── requirements.txt
```

---

## 🏦 Model 1: Prediksi Tabungan (LSTM)

### Deskripsi
Model ini memprediksi **berapa kali lagi pengguna perlu menabung** untuk mencapai target tabungannya. Model menggunakan arsitektur LSTM (Long Short-Term Memory) yang mampu mengenali pola perilaku menabung pengguna dari waktu ke waktu.

### Arsitektur Model
- **Input**: Sekuens 5 langkah waktu (time steps)
- **Layer 1**: Masking → LSTM(128, return_sequences=True) → Dropout(0.2)
- **Layer 2**: LSTM(64) → Dense(32, relu)
- **Output**: Dense(1, linear) — estimasi sisa kali nabung

### Fitur Input
| Fitur | Keterangan |
|---|---|
| `target_nominal` | Nominal target tabungan |
| `nominal_nabung` | Jumlah yang ditabung per sesi |
| `total_terkumpul` | Total dana yang sudah terkumpul |
| `jarak_hari_nabung` | Jarak hari antar setoran |
| `sisa_nominal` | Selisih target dengan total terkumpul |
| `rumus_kalkulator` | Estimasi kasar sisa cicilan |
| `persentase_target` | Persentase pencapaian target |

### Training
- **Optimizer**: Adam (lr=0.001, dengan ReduceLR otomatis)
- **Loss**: Mean Squared Error (MSE)
- **Epochs**: Maks 100, dengan Early Stopping (patience=15)
- **Batch Size**: 32
- **Split**: GroupShuffleSplit (80% train, 20% test, dikelompokkan per `id_tabungan`)

### File Artefak
| File | Keterangan |
|---|---|
| `model_lstm_tabungan.keras` | Model tersimpan lengkap |
| `best_lstm_weights.weights.h5` | Bobot terbaik saat training |
| `scaler_lstm.pkl` | RobustScaler untuk fitur X |
| `scaler_y_lstm.pkl` | RobustScaler untuk target Y |

### Instalasi & Dependensi
```bash
cd ai-model-tabungan
pip install -r requirements.txt
```

```
pandas
numpy
scikit-learn
joblib
tensorflow[and-cuda]
ipykernel
tqdm
livelossplot
seaborn
tensorboard
```

### Contoh Penggunaan
```python
import numpy as np
import joblib
from tensorflow import keras

# Load model dan scaler
model = keras.models.load_model("model_lstm_tabungan.keras")
scaler_x = joblib.load("scaler_lstm.pkl")
scaler_y = joblib.load("scaler_y_lstm.pkl")

SEQ_LENGTH = 5

# Data histori nabung pengguna (contoh: [target, nominal, terkumpul, jarak_hari])
raw_data = [
    [5000000, 500000, 500000, 0, 4500000, 9, 10.0],
    [5000000, 600000, 1100000, 7, 3900000, 7, 22.0],
    [5000000, 200000, 1300000, 30, 3700000, 19, 26.0],
]

# Padding dan scaling
fitur = np.array(raw_data)
fitur_scaled = scaler_x.transform(fitur)

pad_size = SEQ_LENGTH - len(fitur_scaled)
if pad_size > 0:
    padding = np.zeros((pad_size, fitur_scaled.shape[1]))
    fitur_scaled = np.vstack([padding, fitur_scaled])

X_input = fitur_scaled.reshape(1, SEQ_LENGTH, -1)

# Prediksi
pred_scaled = model.predict(X_input)
pred_raw = scaler_y.inverse_transform(pred_scaled).flatten()
sisa_kali_nabung = int(np.clip(np.round(pred_raw[0]), 0, None))

print(f"Estimasi sisa kali menabung: {sisa_kali_nabung} kali")
```

### Monitoring dengan TensorBoard
```bash
tensorboard --logdir=logs/fit
```

---

## 🏷️ Model 2: Klasifikasi Transaksi (LSTM + Attention)

### Deskripsi
Model ini mengklasifikasikan **deskripsi transaksi keuangan** ke dalam kategori yang sesuai secara otomatis. Dilengkapi dengan pipeline LLM (Groq/LLaMA) sebagai normalisasi teks sebelum masuk ke model klasifikasi.

### Kategori yang Didukung
| Kategori | Contoh Transaksi |
|---|---|
| **Belanja** | beli baju, order shopee |
| **Hiburan** | langganan streaming, nonton bioskop |
| **Investasi** | beli saham, top up reksa dana |
| **Kesehatan** | beli obat, bayar dokter |
| **Konsumsi** | makan siang, beli kopi |
| **Lain-lain** | transfer, kas bon |
| **Pendapatan** | gaji, freelance, bonus |
| **Tagihan** | bayar listrik, bayar kosan |
| **Transportasi** | bensin, grab, commuter line |

### Arsitektur Model
- **Input**: Deskripsi teks transaksi (max 30 token)
- **Layer**: TextVectorization → Embedding → LSTM → AttentionLayer → Dense
- **Output**: Softmax (9 kelas)

### Pipeline Prediksi
```
Input teks pengguna
       ↓
LLM Normalisasi (Groq / LLaMA-3.3-70b)
       ↓
Preprocessing (lowercase, remove punctuation, slang normalization)
       ↓
Text Vectorization (vocab size: 5000)
       ↓
Model LSTM + Attention
       ↓
Kategori Transaksi
```

### File Artefak
| File | Keterangan |
|---|---|
| `model_fitur_klasifikasi.keras` | Model tersimpan lengkap |
| `label_encoder.pkl` | LabelEncoder untuk 9 kategori |
| `vocabulary.json` | Kosakata tokenizer (5000 token) |

### Instalasi & Dependensi
```bash
cd ai_klasifikasi
pip install -r requirements.txt
pip install Sastrawi groq google-genai
```

### Contoh Penggunaan
```python
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load artefak
model = load_model("model_fitur_klasifikasi.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("vocabulary.json", "r") as f:
    vocab = json.load(f)

# Preprocessing sederhana (tanpa LLM)
def preprocessing(teks):
    import re
    teks = teks.lower()
    teks = re.sub(r'[^\w\s]', "", teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

deskripsi = "bayar tagihan listrik bulan ini"
teks_bersih = preprocessing(deskripsi)

# Prediksi
pred = model.predict([teks_bersih])
kategori = le.inverse_transform([np.argmax(pred)])[0]
print(f"Kategori: {kategori}")
```

---

## 🔗 Keterkaitan dengan Proyek Catatanku

Model-model ini merupakan bagian dari ekosistem **Catatanku** (Capstone Project). Data training untuk model klasifikasi diambil dari:
```
https://github.com/Capstone-Catatanku/Data-Science
https://github.com/Capstone-Catatanku/Data-Science-Tabungan.git
```

---

## 📊 Monitoring & Evaluasi

Kedua notebook menyertakan visualisasi evaluasi model berupa:
- Grafik *Actual vs Predicted* (tabungan)
- Distribusi error / residuals (tabungan)
- Grafik akurasi training & validasi (klasifikasi)
- Confusion matrix (klasifikasi)

---

## ⚠️ Catatan

- Model klasifikasi dikembangkan di Google Colab dan menggunakan `google.colab.userdata` untuk API key. Pastikan menggantinya dengan metode manajemen secret yang sesuai di lingkungan produksi.
- Gunakan GPU untuk inference yang lebih cepat (`tensorflow[and-cuda]`).
- `scaler_lstm.pkl` dan `scaler_y_lstm.pkl` **wajib** digunakan bersamaan dengan model tabungan agar hasil prediksi valid.
