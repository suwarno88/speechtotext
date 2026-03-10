# 🎙️ Simulasi Interaktif: Speech to Text (STT) — v2.0

Simulasi visual dan interaktif proses **Speech to Text** — dari input suara hingga output teks — untuk mata kuliah **Fondasi Kecerdasan Buatan** (Program Studi PGSD).

## ✨ Fitur Utama

### 3 Mode Input Audio
| Mode | Deskripsi |
|------|-----------|
| 🎤 **Rekam Suara** | Rekam langsung dari mikrofon browser, diproses otomatis |
| 📁 **Upload File** | Upload file audio (WAV, MP3, OGG, FLAC, M4A) |
| 📋 **Kalimat Sampel** | Pilih kalimat contoh untuk simulasi tanpa audio |

### Pipeline Visual 5 Tahap
- **Input Audio** — Waveform interaktif dari sinyal asli
- **Pra-Pemrosesan** — Perbandingan sebelum/sesudah noise reduction
- **Ekstraksi Fitur** — Spektrogram + MFCC heatmap
- **Inferensi Model** — Confidence score per kata
- **Output Teks** — Hasil konversi + metrik keseluruhan

### Fitur Tambahan
- 🌐 Speech Recognition via **Google Speech API** (gratis)
- ⚙️ Parameter yang dapat diatur (noise, sample rate, MFCC)
- 🌍 Multi-bahasa: Indonesia, English, Jawa, Sunda
- 📊 Visualisasi Plotly interaktif di setiap tahap

## 🚀 Cara Menjalankan

### Lokal

```bash
git clone https://github.com/USERNAME/simulasi-speech-to-text.git
cd simulasi-speech-to-text

pip install -r requirements.txt

streamlit run app.py
```

> **Catatan:** Untuk fitur upload MP3/OGG/FLAC/M4A, install juga `ffmpeg`:
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - macOS: `brew install ffmpeg`
> - Windows: Download dari https://ffmpeg.org

### Deploy ke Streamlit Cloud

1. Push semua file ke repository GitHub (termasuk folder `.streamlit/`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Hubungkan repository → pilih `app.py` → Deploy
4. File `packages.txt` otomatis menginstall `ffmpeg` di server

## 📁 Struktur File

```
simulasi-speech-to-text/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependensi Python
├── packages.txt           # Dependensi sistem (ffmpeg)
├── .streamlit/
│   └── config.toml        # Konfigurasi tema dark mode
└── README.md              # Dokumentasi
```

### Penjelasan File

| File | Fungsi |
|------|--------|
| `app.py` | Kode utama: UI, audio processing, visualisasi, STT |
| `requirements.txt` | Library Python yang dibutuhkan (`pip install`) |
| `packages.txt` | Software sistem untuk Streamlit Cloud (`apt install`) |
| `.streamlit/config.toml` | Tema warna dark mode agar tampil konsisten |

## 📚 Tahapan yang Disimulasikan

| # | Tahap | Penjelasan |
|---|-------|-----------|
| 1 | **Input Audio** | Perekaman & digitalisasi gelombang suara |
| 2 | **Pra-Pemrosesan** | Noise reduction, normalisasi, silence trimming |
| 3 | **Ekstraksi Fitur** | MFCC (Mel-Frequency Cepstral Coefficients) |
| 4 | **Inferensi Model** | Decoding oleh model AI (simulasi + Google API) |
| 5 | **Output Teks** | Teks hasil konversi + confidence score |

## 🛠️ Teknologi

- **Python 3.8+**
- **Streamlit** — antarmuka web interaktif
- **audio-recorder-streamlit** — komponen rekaman mikrofon
- **SpeechRecognition** — speech-to-text via Google API
- **pydub + ffmpeg** — konversi format audio
- **NumPy** — pemrosesan sinyal digital
- **Plotly** — visualisasi interaktif

---

Dibuat untuk keperluan pendidikan — Fondasi Kecerdasan Buatan, PGSD.
