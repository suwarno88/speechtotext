# 🎙️ Simulasi Interaktif: Speech to Text (STT)

Simulasi visual dan interaktif proses **Speech to Text** — dari input suara hingga output teks — untuk mata kuliah **Fondasi Kecerdasan Buatan** (Program Studi PGSD).

## 📸 Fitur Utama

- **5 tahapan proses STT** divisualisasikan secara bertahap
- **Sinyal audio simulasi** dengan waveform interaktif (Plotly)
- **Spektrogram & MFCC** — visualisasi ekstraksi fitur akustik
- **Confidence score** per kata dari proses decoding
- **Parameter yang dapat diatur**: noise level, sample rate, jumlah MFCC
- **5 kalimat sampel** dari berbagai domain (pendidikan, teknologi, geografi, sains, seni)

## 🚀 Cara Menjalankan

### Lokal

```bash
# Clone repository
git clone https://github.com/USERNAME/simulasi-speech-to-text.git
cd simulasi-speech-to-text

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

### Deploy ke Streamlit Cloud

1. Push repository ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Hubungkan repository → pilih `app.py` → Deploy

## 📁 Struktur File

```
simulasi-speech-to-text/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Daftar dependensi Python
├── .streamlit/
│   └── config.toml        # Konfigurasi tema Streamlit
└── README.md              # Dokumentasi
```

## 📚 Tahapan yang Disimulasikan

| # | Tahap | Penjelasan |
|---|-------|-----------|
| 1 | **Input Audio** | Perekaman & digitalisasi gelombang suara |
| 2 | **Pra-Pemrosesan** | Noise reduction, normalisasi, trimming |
| 3 | **Ekstraksi Fitur** | MFCC (Mel-Frequency Cepstral Coefficients) |
| 4 | **Inferensi Model** | Decoding oleh model AI (simulasi) |
| 5 | **Output Teks** | Teks hasil konversi + confidence score |

## 🛠️ Teknologi

- **Python 3.8+**
- **Streamlit** — antarmuka web interaktif
- **NumPy** — pemrosesan sinyal
- **Plotly** — visualisasi interaktif

---

Dibuat untuk keperluan pendidikan — Fondasi Kecerdasan Buatan, PGSD.
