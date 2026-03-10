"""
=================================================================
SIMULASI INTERAKTIF: SPEECH TO TEXT (STT) — v2.0
Fondasi Kecerdasan Buatan — Program Studi PGSD
=================================================================
Fitur:
  ✅ Rekam suara langsung dari mikrofon browser
  ✅ Upload file audio (WAV, MP3, OGG, FLAC, M4A)
  ✅ Pilih kalimat sampel (simulasi)
  ✅ Pipeline visual 5 tahap dengan visualisasi Plotly
  ✅ Speech Recognition via Google Speech API

Dibuat dengan: Python + Streamlit + SpeechRecognition
=================================================================
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import wave
import struct

# -- Library untuk audio recording & speech recognition --
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr_lib

# =================================================================
# KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(
    page_title="Simulasi Speech to Text",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# CUSTOM CSS
# =================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

.stApp { font-family: 'DM Sans', sans-serif; }

/* Stage cards */
.stage-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    transition: all 0.3s;
}
.stage-card:hover { border-color: #22d3ee; }
.stage-card.active {
    border-color: #22d3ee;
    box-shadow: 0 0 24px rgba(34,211,238,0.15);
}

.stage-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 12px;
}
.stage-num {
    width: 32px; height: 32px; border-radius: 10px;
    background: rgba(34,211,238,0.15);
    color: #22d3ee; font-weight: 700; font-size: 0.85rem;
    display: grid; place-items: center;
}
.stage-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }
.stage-desc { font-size: 0.88rem; color: #94a3b8; line-height: 1.7; }

/* Info box */
.info-box {
    background: rgba(34,211,238,0.08);
    border-left: 3px solid #22d3ee;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 12px 0;
    font-size: 0.88rem; color: #cbd5e1; line-height: 1.7;
}

/* Warning box */
.warn-box {
    background: rgba(251,191,36,0.08);
    border-left: 3px solid #fbbf24;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 12px 0;
    font-size: 0.88rem; color: #fbbf24; line-height: 1.7;
}

/* Result box */
.result-box {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 14px;
    padding: 20px 24px; margin: 12px 0;
    font-size: 1.15rem; color: #34d399;
    text-align: center; font-weight: 600;
}

/* Pipeline */
.pipeline-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 0; gap: 4px; flex-wrap: wrap;
}
.pipe-node {
    text-align: center; padding: 10px 8px;
    border-radius: 12px; flex: 1; min-width: 100px;
    background: rgba(255,255,255,0.03);
    border: 1px solid #1e293b;
    transition: all 0.3s;
}
.pipe-node.done {
    background: rgba(34,211,238,0.1);
    border-color: #22d3ee;
}
.pipe-node.current {
    background: rgba(251,191,36,0.1);
    border-color: #fbbf24;
    animation: glow 1.5s ease infinite alternate;
}
@keyframes glow {
    from { box-shadow: 0 0 4px rgba(251,191,36,0.2); }
    to { box-shadow: 0 0 16px rgba(251,191,36,0.4); }
}
.pipe-icon { font-size: 1.4rem; }
.pipe-label { font-size: 0.7rem; color: #94a3b8; margin-top: 4px; }
.pipe-arrow { color: #334155; font-size: 1rem; flex-shrink: 0; }

/* Metric card */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px; text-align: center;
}
.metric-val { font-size: 1.4rem; font-weight: 700; color: #22d3ee; }
.metric-label { font-size: 0.75rem; color: #64748b; margin-top: 4px; }

/* Input mode tabs */
.mode-tab {
    background: rgba(255,255,255,0.04);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 24px; margin: 8px 0;
    text-align: center;
}
.mode-tab h4 { color: #e2e8f0; margin-bottom: 8px; }
.mode-tab p { color: #94a3b8; font-size: 0.85rem; }

/* Audio source badge */
.source-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px 0;
}
.source-badge.mic { background: rgba(244,114,182,0.15); color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }
.source-badge.upload { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.source-badge.sample { background: rgba(34,211,238,0.15); color: #22d3ee; border: 1px solid rgba(34,211,238,0.3); }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =================================================================
# DATA SAMPEL
# =================================================================
SAMPLE_SENTENCES = {
    "🎓 Kalimat Pendidikan": {
        "text": "Guru adalah pahlawan tanpa tanda jasa yang mendidik generasi bangsa",
        "duration": 3.8,
        "words": ["Guru", "adalah", "pahlawan", "tanpa", "tanda", "jasa",
                   "yang", "mendidik", "generasi", "bangsa"],
    },
    "🤖 Kalimat Teknologi": {
        "text": "Kecerdasan buatan membantu manusia menyelesaikan tugas secara efisien",
        "duration": 3.5,
        "words": ["Kecerdasan", "buatan", "membantu", "manusia",
                   "menyelesaikan", "tugas", "secara", "efisien"],
    },
    "🌍 Kalimat Geografi": {
        "text": "Indonesia adalah negara kepulauan terbesar di dunia dengan ribuan pulau",
        "duration": 4.0,
        "words": ["Indonesia", "adalah", "negara", "kepulauan", "terbesar",
                   "di", "dunia", "dengan", "ribuan", "pulau"],
    },
    "📚 Kalimat Sains": {
        "text": "Fotosintesis mengubah cahaya matahari menjadi energi untuk tumbuhan",
        "duration": 3.2,
        "words": ["Fotosintesis", "mengubah", "cahaya", "matahari",
                   "menjadi", "energi", "untuk", "tumbuhan"],
    },
    "🎵 Kalimat Seni": {
        "text": "Musik tradisional Indonesia memiliki keragaman yang sangat kaya",
        "duration": 3.0,
        "words": ["Musik", "tradisional", "Indonesia", "memiliki",
                   "keragaman", "yang", "sangat", "kaya"],
    },
}


# =================================================================
# FUNGSI: AUDIO PROCESSING
# =================================================================

def read_audio_bytes(audio_bytes):
    """
    Membaca audio bytes (WAV) dan mengekstrak sinyal + sample rate.
    Mendukung output dari audio_recorder (format WAV) dan file upload.
    """
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Konversi raw bytes ke numpy array
        if sampwidth == 1:
            fmt = f"{n_frames * n_channels}B"
            data = np.array(struct.unpack(fmt, raw), dtype=np.float32) - 128.0
            data /= 128.0
        elif sampwidth == 2:
            fmt = f"{n_frames * n_channels}h"
            data = np.array(struct.unpack(fmt, raw), dtype=np.float32)
            data /= 32768.0
        elif sampwidth == 4:
            fmt = f"{n_frames * n_channels}i"
            data = np.array(struct.unpack(fmt, raw), dtype=np.float32)
            data /= 2147483648.0
        else:
            return None, None, None

        # Jika stereo, ambil rata-rata channel
        if n_channels == 2:
            data = (data[0::2] + data[1::2]) / 2.0

        duration = len(data) / framerate
        t = np.linspace(0, duration, len(data))
        return t, data, framerate

    except Exception:
        return None, None, None


def convert_audio_to_wav(audio_bytes, original_format):
    """
    Mengonversi file audio ke WAV menggunakan pydub jika tersedia,
    atau fallback ke raw bytes jika sudah WAV.
    """
    if original_format in ['wav', 'wave']:
        return audio_bytes

    try:
        from pydub import AudioSegment
        buf = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(buf, format=original_format)
        # Konversi ke WAV mono 16kHz 16bit
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        wav_buf = io.BytesIO()
        audio.export(wav_buf, format="wav")
        return wav_buf.getvalue()
    except ImportError:
        return None
    except Exception:
        return None


def recognize_speech(audio_bytes, language="id-ID"):
    """
    Menjalankan speech recognition menggunakan Google Speech API.
    Input: audio bytes dalam format WAV.
    Output: (teks, confidence) atau (None, None) jika gagal.
    """
    recognizer = sr_lib.Recognizer()
    try:
        buf = io.BytesIO(audio_bytes)
        with sr_lib.AudioFile(buf) as source:
            audio_data = recognizer.record(source)

        # Gunakan Google Speech API (gratis, tanpa API key)
        result = recognizer.recognize_google(
            audio_data,
            language=language,
            show_all=True
        )

        if result and isinstance(result, dict) and 'alternative' in result:
            best = result['alternative'][0]
            text = best.get('transcript', '')
            conf = best.get('confidence', 0.85)
            return text, conf
        elif isinstance(result, str):
            return result, 0.85
        else:
            return None, None

    except sr_lib.UnknownValueError:
        return None, None
    except sr_lib.RequestError as e:
        st.error(f"⚠️ Gagal terhubung ke Google Speech API: {e}")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error saat recognition: {e}")
        return None, None


def text_to_sentence_data(text, duration=None):
    """Mengonversi teks hasil STT menjadi format sentence_data."""
    words = text.strip().split()
    if not words:
        return None
    est_duration = duration if duration else max(1.5, len(words) * 0.4)
    return {
        "text": text.strip(),
        "duration": round(est_duration, 1),
        "words": words,
    }


# =================================================================
# FUNGSI: GENERATE SINYAL AUDIO SIMULASI
# =================================================================
def generate_raw_audio(sentence_data, sr=16000):
    """Menghasilkan sinyal audio simulasi dari data kalimat."""
    duration = sentence_data["duration"]
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)

    n_words = len(sentence_data["words"])
    word_dur = duration / n_words

    for i, word in enumerate(sentence_data["words"]):
        f0 = 100 + len(word) * 15
        start = int(i * word_dur * sr)
        end = int((i + 1) * word_dur * sr)
        seg_t = t[start:end]

        seg = (0.5 * np.sin(2 * np.pi * f0 * seg_t) +
               0.3 * np.sin(2 * np.pi * f0 * 2 * seg_t) +
               0.15 * np.sin(2 * np.pi * f0 * 3 * seg_t))

        env = np.ones(len(seg))
        fade = min(int(0.05 * sr), len(seg) // 4)
        if fade > 0:
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
        seg *= env
        signal[start:end] = seg

    noise = np.random.normal(0, 0.05, len(signal))
    signal += noise
    return t, signal, sr


def preprocess_audio(t, signal, sr):
    """Pra-pemrosesan: noise reduction + normalisasi."""
    window = 5
    padded = np.pad(signal, (window // 2, window // 2), mode='edge')
    clean = np.convolve(padded, np.ones(window) / window, mode='valid')[:len(signal)]

    peak = np.max(np.abs(clean))
    if peak > 0:
        clean = clean / peak

    threshold = 0.05
    active = np.abs(clean) > threshold
    if np.any(active):
        first = np.argmax(active)
        last = len(active) - np.argmax(active[::-1])
        trimmed = clean[first:last]
        t_trimmed = t[first:last]
    else:
        trimmed, t_trimmed = clean, t

    return t_trimmed, trimmed


def extract_features(signal, sr, n_filters=26, n_mfcc=13):
    """Ekstraksi fitur MFCC (simulasi sederhana)."""
    frame_size = int(0.025 * sr)
    hop = int(0.01 * sr)
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)

    mel_energies = np.zeros((n_frames, n_filters))
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_size, len(signal))
        frame = signal[start:end]

        if len(frame) > 0:
            spectrum = np.abs(np.fft.rfft(frame, n=frame_size)) ** 2
            filter_len = len(spectrum) // n_filters
            for j in range(n_filters):
                s = j * filter_len
                e = min(s + filter_len, len(spectrum))
                mel_energies[i, j] = np.mean(spectrum[s:e]) if e > s else 0

    mel_energies = np.maximum(mel_energies, 1e-10)
    log_mel = np.log(mel_energies)

    mfcc = np.zeros((n_frames, n_mfcc))
    for i in range(n_mfcc):
        for j in range(n_filters):
            mfcc[:, i] += log_mel[:, j] * np.cos(np.pi * i * (2 * j + 1) / (2 * n_filters))

    return mfcc, mel_energies, log_mel


def simulate_decoding(sentence_data, mfcc):
    """Simulasi proses decoding (inferensi model)."""
    words = sentence_data["words"]
    n_frames = mfcc.shape[0]
    frames_per_word = max(1, n_frames // len(words))

    results = []
    for i, word in enumerate(words):
        base_conf = 0.82 + np.random.random() * 0.16
        if len(word) <= 3:
            base_conf *= 0.95

        frame_start = i * frames_per_word
        frame_end = min((i + 1) * frames_per_word, n_frames)

        alternatives = []
        if len(word) > 4:
            alt = word[:len(word)//2] + word[len(word)//2:].replace('a', 'e', 1)
            alternatives.append({"text": alt, "score": base_conf * 0.7})

        results.append({
            "word": word,
            "confidence": round(base_conf, 3),
            "frame_range": (frame_start, frame_end),
            "alternatives": alternatives
        })

    overall_conf = np.mean([r["confidence"] for r in results])
    return results, overall_conf


# =================================================================
# FUNGSI VISUALISASI
# =================================================================
def plot_waveform(t, signal, title="Sinyal Audio", color="#22d3ee"):
    fig = go.Figure()
    # Downsample untuk performa jika sinyal terlalu panjang
    max_points = 8000
    if len(t) > max_points:
        step = len(t) // max_points
        t_plot, s_plot = t[::step], signal[::step]
    else:
        t_plot, s_plot = t, signal

    fig.add_trace(go.Scatter(
        x=t_plot, y=s_plot, mode='lines',
        line=dict(color=color, width=1.2),
        fill='tozeroy', fillcolor='rgba(34,211,238,0.08)',
        hovertemplate='Waktu: %{x:.3f}s<br>Amplitudo: %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(title="Waktu (detik)", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Amplitudo", gridcolor="#1e293b", color="#94a3b8"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=50, r=20, t=40, b=40),
        font=dict(family="DM Sans", color="#94a3b8")
    )
    return fig


def plot_comparison(t1, s1, t2, s2):
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Sinyal Mentah (Raw)", "Setelah Preprocessing"),
                        vertical_spacing=0.15)
    for tx, sx, row, color in [(t1, s1, 1, "#fb923c"), (t2, s2, 2, "#22d3ee")]:
        max_pts = 5000
        if len(tx) > max_pts:
            step = len(tx) // max_pts
            tx, sx = tx[::step], sx[::step]
        fig.add_trace(go.Scatter(x=tx, y=sx, mode='lines',
                                  line=dict(color=color, width=1)), row=row, col=1)

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=50, r=20, t=40, b=30), showlegend=False,
        font=dict(family="DM Sans", color="#94a3b8", size=11)
    )
    fig.update_xaxes(gridcolor="#1e293b", color="#64748b")
    fig.update_yaxes(gridcolor="#1e293b", color="#64748b")
    return fig


def plot_spectrogram(mel_energies):
    fig = go.Figure(data=go.Heatmap(
        z=mel_energies.T,
        colorscale=[[0, '#0f172a'], [0.3, '#1e3a5f'], [0.5, '#0e7490'],
                     [0.7, '#22d3ee'], [0.9, '#fbbf24'], [1.0, '#f472b6']],
        hovertemplate='Frame: %{x}<br>Filter: %{y}<br>Energi: %{z:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Mel Filterbank Energi (Spektrogram Simulasi)", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Frame", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Mel Filter", gridcolor="#1e293b", color="#94a3b8"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=50, r=20, t=40, b=40),
        font=dict(family="DM Sans", color="#94a3b8")
    )
    return fig


def plot_mfcc(mfcc):
    fig = go.Figure(data=go.Heatmap(
        z=mfcc.T,
        colorscale=[[0, '#1e1b4b'], [0.25, '#312e81'], [0.5, '#4f46e5'],
                     [0.75, '#818cf8'], [1.0, '#c7d2fe']],
        hovertemplate='Frame: %{x}<br>MFCC-%{y}<br>Nilai: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=f"Koefisien MFCC ({mfcc.shape[1]} Koefisien)", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Frame", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="MFCC Index", gridcolor="#1e293b", color="#94a3b8"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=50, r=20, t=40, b=40),
        font=dict(family="DM Sans", color="#94a3b8")
    )
    return fig


def plot_confidence(decode_results):
    words = [r["word"] for r in decode_results]
    confs = [r["confidence"] * 100 for r in decode_results]
    colors = ['#34d399' if c >= 90 else '#fbbf24' if c >= 80 else '#f87171' for c in confs]

    fig = go.Figure(go.Bar(
        x=words, y=confs, marker_color=colors,
        text=[f"{c:.1f}%" for c in confs], textposition='outside',
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Confidence Score per Kata", font=dict(size=13, color="#e2e8f0")),
        xaxis=dict(title="Kata", gridcolor="#1e293b", color="#94a3b8", tickangle=-30),
        yaxis=dict(title="Confidence (%)", gridcolor="#1e293b", color="#94a3b8", range=[0, 105]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=50, r=20, t=40, b=60),
        font=dict(family="DM Sans", color="#94a3b8")
    )
    return fig


def plot_pipeline_diagram():
    fig = go.Figure()
    stages = [
        {"x": 0, "icon": "🎙️", "label": "Input\nAudio", "color": "#fb923c"},
        {"x": 1, "icon": "🔧", "label": "Pra-\nPemrosesan", "color": "#a78bfa"},
        {"x": 2, "icon": "📊", "label": "Ekstraksi\nFitur", "color": "#22d3ee"},
        {"x": 3, "icon": "🧠", "label": "Inferensi\nModel", "color": "#f472b6"},
        {"x": 4, "icon": "📝", "label": "Output\nTeks", "color": "#34d399"},
    ]
    for s in stages:
        fig.add_shape(type="rect", x0=s["x"]-0.35, x1=s["x"]+0.35, y0=-0.4, y1=0.4,
                       fillcolor=s["color"], opacity=0.15, line=dict(color=s["color"], width=2),
                       layer="below")
        fig.add_annotation(x=s["x"], y=0.15, text=s["icon"], showarrow=False, font=dict(size=24))
        fig.add_annotation(x=s["x"], y=-0.2, text=s["label"], showarrow=False,
                            font=dict(size=10, color="#e2e8f0"), align="center")
    for i in range(len(stages) - 1):
        fig.add_annotation(x=stages[i]["x"]+0.42, y=0, ax=stages[i+1]["x"]-0.42, ay=0,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.2,
                            arrowcolor="#475569", arrowwidth=2)
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.8, 4.8]),
        yaxis=dict(visible=False, range=[-0.7, 0.7]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=160, margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


# =================================================================
# FUNGSI RENDER PIPELINE STATUS
# =================================================================
def render_pipeline_status(current_step):
    steps = [
        ("🎙️", "Input"), ("🔧", "Preprocessing"),
        ("📊", "Fitur"), ("🧠", "Inferensi"), ("📝", "Output"),
    ]
    html = '<div class="pipeline-bar">'
    for i, (icon, label) in enumerate(steps):
        cls = "done" if i < current_step else ("current" if i == current_step else "")
        html += f'<div class="pipe-node {cls}"><div class="pipe-icon">{icon}</div><div class="pipe-label">{label}</div></div>'
        if i < len(steps) - 1:
            html += '<div class="pipe-arrow">→</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# =================================================================
# FUNGSI: RUN FULL PIPELINE (dipakai oleh semua mode input)
# =================================================================
def run_full_pipeline(sentence_data, t_audio, signal_audio, actual_sr,
                      noise_level, target_sr, n_mfcc, source_label, stt_confidence=None):
    """
    Menjalankan dan menampilkan seluruh 5 tahap pipeline STT.
    """

    # ============================================================
    # TAHAP 1: INPUT AUDIO
    # ============================================================
    st.markdown('<a id="tahap-1-input-audio"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stage-card active">
        <div class="stage-header">
            <div class="stage-num">1</div>
            <div class="stage-title">🎙️ Tahap 1: Input Audio — Perekaman Suara</div>
        </div>
        <div class="stage-desc">
            Mikrofon menangkap gelombang suara dan mengubahnya menjadi sinyal digital melalui proses
            <strong>sampling</strong> — mengambil ribuan titik data per detik. Semakin tinggi sample rate,
            semakin detail suara yang terekam.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_pipeline_status(0)

    # Badge sumber audio
    badge_cls = {"mic": "mic", "upload": "upload", "sample": "sample"}
    badge_text = {"mic": "🎤 Rekaman Mikrofon", "upload": "📁 File Upload", "sample": "📋 Kalimat Sampel"}
    st.markdown(f'<span class="source-badge {badge_cls[source_label]}">{badge_text[source_label]}</span>',
                unsafe_allow_html=True)

    st.plotly_chart(plot_waveform(t_audio, signal_audio, "Gelombang Suara Mentah (Raw Audio)", "#fb923c"),
                     use_container_width=True, config={'displayModeBar': False})

    duration = t_audio[-1] - t_audio[0] if len(t_audio) > 1 else 0
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{actual_sr:,}</div><div class="metric-label">Sample Rate (Hz)</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{len(signal_audio):,}</div><div class="metric-label">Total Sampel</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{duration:.1f}s</div><div class="metric-label">Durasi Audio</div></div>', unsafe_allow_html=True)

    st.success("✅ Audio berhasil dimuat!")
    st.markdown("---")

    # ============================================================
    # TAHAP 2: PRA-PEMROSESAN
    # ============================================================
    st.markdown('<a id="tahap-2-pra-pemrosesan"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stage-card active">
        <div class="stage-header">
            <div class="stage-num">2</div>
            <div class="stage-title">🔧 Tahap 2: Pra-Pemrosesan Audio</div>
        </div>
        <div class="stage-desc">
            Sinyal mentah dibersihkan dari <strong>noise</strong> (suara latar belakang),
            <strong>dinormalisasi</strong> volumenya agar konsisten, dan bagian hening (silence)
            dipotong. Hasilnya: sinyal bersih yang siap diekstrak fiturnya.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_pipeline_status(1)

    with st.spinner("🔧 Membersihkan noise & normalisasi..."):
        time.sleep(0.8)
        t_clean, signal_clean = preprocess_audio(t_audio, signal_audio, actual_sr)

    st.plotly_chart(plot_comparison(t_audio, signal_audio, t_clean, signal_clean),
                     use_container_width=True, config={'displayModeBar': False})

    noise_reduction = (1 - np.std(signal_clean) / max(np.std(signal_audio), 1e-10)) * 100
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{noise_reduction:.1f}%</div><div class="metric-label">Noise Reduction</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{np.max(np.abs(signal_clean)):.2f}</div><div class="metric-label">Peak Amplitude</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{len(signal_clean):,}</div><div class="metric-label">Sampel Setelah Trim</div></div>', unsafe_allow_html=True)

    st.success("✅ Pra-pemrosesan selesai!")
    st.markdown("---")

    # ============================================================
    # TAHAP 3: EKSTRAKSI FITUR
    # ============================================================
    st.markdown('<a id="tahap-3-ekstraksi-fitur"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stage-card active">
        <div class="stage-header">
            <div class="stage-num">3</div>
            <div class="stage-title">📊 Tahap 3: Ekstraksi Fitur Audio (MFCC)</div>
        </div>
        <div class="stage-desc">
            Sinyal dipecah menjadi <strong>frame</strong> pendek (25ms), lalu setiap frame diekstrak
            fitur akustiknya. Hasilnya berupa <strong>MFCC</strong> (Mel-Frequency Cepstral Coefficients)
            — representasi matematis yang menangkap karakteristik penting suara.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_pipeline_status(2)

    with st.spinner("📊 Mengekstrak fitur MFCC..."):
        time.sleep(0.8)
        mfcc, mel_energies, log_mel = extract_features(signal_clean, actual_sr, n_mfcc=n_mfcc)

    tab1, tab2 = st.tabs(["🔥 Spektrogram (Mel Filterbank)", "📈 Koefisien MFCC"])
    with tab1:
        st.plotly_chart(plot_spectrogram(mel_energies), use_container_width=True, config={'displayModeBar': False})
    with tab2:
        st.plotly_chart(plot_mfcc(mfcc), use_container_width=True, config={'displayModeBar': False})

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{mfcc.shape[0]}</div><div class="metric-label">Jumlah Frame</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{n_mfcc}</div><div class="metric-label">Koefisien MFCC</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{mfcc.shape[0] * n_mfcc:,}</div><div class="metric-label">Total Fitur</div></div>', unsafe_allow_html=True)

    st.success("✅ Ekstraksi fitur selesai!")
    st.markdown("---")

    # ============================================================
    # TAHAP 4: INFERENSI MODEL
    # ============================================================
    st.markdown('<a id="tahap-4-inferensi-model"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stage-card active">
        <div class="stage-header">
            <div class="stage-num">4</div>
            <div class="stage-title">🧠 Tahap 4: Inferensi Model (Decoding)</div>
        </div>
        <div class="stage-desc">
            Model AI (biasanya <strong>deep learning</strong> seperti RNN/Transformer) membandingkan
            fitur MFCC dengan pola yang dipelajari dari jutaan data latih. Proses <strong>beam search</strong>
            digunakan untuk menemukan urutan kata yang paling mungkin cocok.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_pipeline_status(3)

    with st.spinner("🧠 Model AI sedang mencocokkan pola..."):
        time.sleep(1.0)
        decode_results, overall_conf = simulate_decoding(sentence_data, mfcc)

    # Jika ada confidence dari STT asli, gunakan sebagai faktor
    if stt_confidence is not None:
        overall_conf = stt_confidence

    st.plotly_chart(plot_confidence(decode_results), use_container_width=True, config={'displayModeBar': False})

    with st.expander("📋 Detail Hasil Decoding per Kata", expanded=False):
        for i, r in enumerate(decode_results):
            conf_color = "#34d399" if r["confidence"] >= 0.90 else "#fbbf24" if r["confidence"] >= 0.80 else "#f87171"
            st.markdown(f"""
            **Kata {i+1}: `{r['word']}`** — Confidence:
            <span style="color:{conf_color}; font-weight:700;">{r['confidence']*100:.1f}%</span>
            &nbsp;|&nbsp; Frame: {r['frame_range'][0]}–{r['frame_range'][1]}
            """, unsafe_allow_html=True)

    st.success("✅ Inferensi model selesai!")
    st.markdown("---")

    # ============================================================
    # TAHAP 5: OUTPUT TEKS
    # ============================================================
    st.markdown('<a id="tahap-5-output-teks"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stage-card active">
        <div class="stage-header">
            <div class="stage-num">5</div>
            <div class="stage-title">📝 Tahap 5: Output Teks — Hasil Akhir</div>
        </div>
        <div class="stage-desc">
            Semua kata yang berhasil didekode disusun menjadi kalimat utuh. Sistem juga memberikan
            <strong>confidence score</strong> keseluruhan yang menunjukkan seberapa yakin model
            terhadap hasil konversi.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_pipeline_status(4)

    st.markdown(f"""
    <div class="result-box">
        🎯 "{sentence_data['text']}"
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{overall_conf*100:.1f}%</div><div class="metric-label">Confidence Keseluruhan</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{len(sentence_data["words"])}</div><div class="metric-label">Kata Terdeteksi</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{len(sentence_data["text"])}</div><div class="metric-label">Karakter</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-val">{duration:.1f}s</div><div class="metric-label">Durasi Audio</div></div>', unsafe_allow_html=True)

    st.balloons()

    # ---- Ringkasan ----
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>🎉 Simulasi Selesai!</strong><br><br>
        Anda telah melihat 5 tahap utama proses <strong>Speech to Text</strong>:<br>
        1️⃣ <strong>Input Audio</strong> — Suara ditangkap dan didigitalisasi<br>
        2️⃣ <strong>Pra-Pemrosesan</strong> — Noise dihapus, sinyal dinormalisasi<br>
        3️⃣ <strong>Ekstraksi Fitur</strong> — MFCC diekstrak dari sinyal<br>
        4️⃣ <strong>Inferensi Model</strong> — AI mencocokkan pola ke kata<br>
        5️⃣ <strong>Output Teks</strong> — Teks akhir ditampilkan<br><br>
        💡 <em>Coba mode input lain atau ubah parameter di sidebar untuk eksplorasi lebih lanjut!</em>
    </div>
    """, unsafe_allow_html=True)


# =================================================================
# SIDEBAR
# =================================================================
with st.sidebar:
    st.markdown("## 🎙️ Simulasi STT v2")
    st.markdown("---")

    st.markdown("### 🔊 Mode Input Audio")
    input_mode = st.radio(
        "Pilih sumber audio:",
        ["📋 Kalimat Sampel", "🎤 Rekam Suara", "📁 Upload File Audio"],
        index=0,
        help="Pilih bagaimana audio akan dimasukkan ke sistem."
    )

    # -- Mode: Kalimat Sampel --
    if input_mode == "📋 Kalimat Sampel":
        st.markdown("---")
        st.markdown("### 📋 Pilih Kalimat")
        selected_key = st.radio(
            "Pilih kalimat:",
            list(SAMPLE_SENTENCES.keys()),
            index=0,
            label_visibility="collapsed"
        )

    # -- Mode: Rekam Suara --
    elif input_mode == "🎤 Rekam Suara":
        st.markdown("---")
        st.markdown("### 🎤 Rekam Suara Anda")
        st.markdown("""
        <div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;">
        Klik ikon mikrofon di bawah untuk mulai merekam.
        Klik lagi untuk berhenti. Audio akan diproses <strong>otomatis</strong>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        # Komponen audio recorder di sidebar
        audio_bytes_recorded = audio_recorder(
            text="",
            recording_color="#f472b6",
            neutral_color="#22d3ee",
            icon_size="2x",
            pause_threshold=2.5,
            sample_rate=16000
        )

    # -- Mode: Upload File --
    elif input_mode == "📁 Upload File Audio":
        st.markdown("---")
        st.markdown("### 📁 Upload File Audio")
        uploaded_file = st.file_uploader(
            "Pilih file audio",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help="Format didukung: WAV, MP3, OGG, FLAC, M4A. Rekomendasi: WAV mono < 30 detik."
        )

    st.markdown("---")
    st.markdown("### ⚙️ Parameter Simulasi")

    noise_level = st.slider("🔊 Level Noise", 0.01, 0.15, 0.05, 0.01,
                             help="Tingkat noise pada sinyal audio")
    sample_rate = st.select_slider("📐 Sample Rate (Hz)",
                                    options=[8000, 16000, 22050, 44100],
                                    value=16000,
                                    help="Jumlah sampel per detik")
    n_mfcc = st.slider("📊 Jumlah MFCC", 6, 20, 13,
                         help="Jumlah koefisien MFCC yang diekstrak")
    stt_language = st.selectbox("🌐 Bahasa STT",
                                 ["id-ID", "en-US", "en-GB", "jv-ID", "su-ID"],
                                 index=0,
                                 help="Bahasa untuk speech recognition (mode Rekam & Upload)")

    st.markdown("---")
    st.markdown("### 📖 Navigasi Tahap")
    st.markdown("""
    - [🎙️ Input Audio](#tahap-1-input-audio)
    - [🔧 Pra-Pemrosesan](#tahap-2-pra-pemrosesan)
    - [📊 Ekstraksi Fitur](#tahap-3-ekstraksi-fitur)
    - [🧠 Inferensi Model](#tahap-4-inferensi-model)
    - [📝 Output Teks](#tahap-5-output-teks)
    """)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#64748b;text-align:center;">
    Dibuat untuk mata kuliah<br>
    <strong style="color:#94a3b8;">Fondasi Kecerdasan Buatan</strong><br>
    Program Studi PGSD
    </div>
    """, unsafe_allow_html=True)


# =================================================================
# KONTEN UTAMA
# =================================================================

# ---- Title ----
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px;">
    <h1 style="font-size:2.2rem; font-weight:800;
        background: linear-gradient(135deg, #22d3ee, #a78bfa, #f472b6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin-bottom:8px;">
        Simulasi Proses Speech to Text
    </h1>
    <p style="color:#94a3b8; font-size:0.95rem; max-width:700px; margin:0 auto; line-height:1.7;">
        Eksplorasi interaktif bagaimana komputer mengubah suara manusia menjadi teks digital —
        gunakan mikrofon, upload file audio, atau pilih kalimat sampel.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- Pipeline Diagram ----
st.plotly_chart(plot_pipeline_diagram(), use_container_width=True, config={'displayModeBar': False})


# =================================================================
# ROUTING BERDASARKAN MODE INPUT
# =================================================================

# ---------------------------------------------------------------
# MODE 1: KALIMAT SAMPEL
# ---------------------------------------------------------------
if input_mode == "📋 Kalimat Sampel":
    sample = SAMPLE_SENTENCES[selected_key]

    st.markdown(f"""
    <div class="info-box">
        📌 <strong>Mode:</strong> Kalimat Sampel &nbsp;|&nbsp;
        <strong>Kalimat:</strong> "{sample['text']}" &nbsp;|&nbsp;
        ⏱️ {sample['duration']}s &nbsp;|&nbsp; 🔤 {len(sample['words'])} kata
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        run_sim = st.button("▶️  Jalankan Simulasi", use_container_width=True, type="primary")

    if run_sim:
        with st.spinner("🎙️ Memproses audio simulasi..."):
            time.sleep(0.8)
            t_raw, signal_raw, sr_val = generate_raw_audio(sample, sr=sample_rate)
            signal_raw += np.random.normal(0, noise_level, len(signal_raw))

        run_full_pipeline(
            sentence_data=sample,
            t_audio=t_raw, signal_audio=signal_raw, actual_sr=sr_val,
            noise_level=noise_level, target_sr=sample_rate, n_mfcc=n_mfcc,
            source_label="sample"
        )
    else:
        st.markdown("")
        st.markdown("""
        <div class="info-box">
            👆 Klik <strong>"Jalankan Simulasi"</strong> untuk memulai pipeline STT
            dengan kalimat sampel yang dipilih di sidebar.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("### 🔍 Apa itu Speech to Text?")
        st.markdown("""
        **Speech to Text (STT)** adalah teknologi yang mengubah suara manusia menjadi teks digital.
        Teknologi ini digunakan di berbagai aplikasi sehari-hari seperti asisten virtual
        (Siri, Google Assistant), transkripsi otomatis, subtitle video, dan lainnya.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 📌 Tahapan Utama
            1. **Input Audio** — Perekaman gelombang suara
            2. **Pra-Pemrosesan** — Pembersihan sinyal
            3. **Ekstraksi Fitur** — Konversi ke representasi matematis
            4. **Inferensi Model** — Pencocokan pola oleh AI
            5. **Output Teks** — Hasil konversi akhir
            """)
        with col2:
            st.markdown("""
            #### 🛠️ Teknologi yang Digunakan
            - **Sampling** — Digitalisasi gelombang suara
            - **Noise Reduction** — Pembersihan noise
            - **MFCC** — Fitur akustik standar industri
            - **Deep Learning** — Neural network untuk decoding
            - **Beam Search** — Algoritma pencarian kata terbaik
            """)


# ---------------------------------------------------------------
# MODE 2: REKAM SUARA
# ---------------------------------------------------------------
elif input_mode == "🎤 Rekam Suara":

    st.markdown("""
    <div class="info-box">
        🎤 <strong>Mode: Rekam Suara</strong> — Gunakan tombol mikrofon di <strong>sidebar kiri</strong>
        untuk merekam suara Anda. Setelah selesai merekam, pipeline akan berjalan <strong>otomatis</strong>.<br>
        💡 Pastikan browser mengizinkan akses mikrofon dan bicara dengan jelas dalam bahasa yang dipilih.
    </div>
    """, unsafe_allow_html=True)

    # Cek apakah ada hasil rekaman
    if 'audio_bytes_recorded' in dir() and audio_bytes_recorded:
        st.markdown("""
        <div class="warn-box">
            ✅ <strong>Audio berhasil direkam!</strong> Memproses speech recognition dan menjalankan pipeline...
        </div>
        """, unsafe_allow_html=True)

        # Playback audio yang direkam
        st.audio(audio_bytes_recorded, format="audio/wav")

        # --- Speech Recognition ---
        with st.spinner("🧠 Menjalankan Speech Recognition (Google API)..."):
            recognized_text, stt_conf = recognize_speech(audio_bytes_recorded, language=stt_language)

        if recognized_text:
            st.markdown(f"""
            <div class="result-box" style="margin-bottom:16px;">
                🗣️ Teks terdeteksi: "{recognized_text}"
            </div>
            """, unsafe_allow_html=True)

            # Parse audio bytes ke sinyal untuk visualisasi
            t_audio, signal_audio, audio_sr = read_audio_bytes(audio_bytes_recorded)
            sentence_data = text_to_sentence_data(
                recognized_text,
                duration=(t_audio[-1] if t_audio is not None and len(t_audio) > 0 else None)
            )

            if t_audio is not None and sentence_data:
                # Gunakan sinyal audio ASLI dari rekaman
                run_full_pipeline(
                    sentence_data=sentence_data,
                    t_audio=t_audio, signal_audio=signal_audio, actual_sr=audio_sr,
                    noise_level=noise_level, target_sr=sample_rate, n_mfcc=n_mfcc,
                    source_label="mic", stt_confidence=stt_conf
                )
            elif sentence_data:
                # Fallback: gunakan sinyal simulasi jika WAV tidak bisa dibaca
                st.markdown("""
                <div class="warn-box">
                    ⚠️ Format audio tidak dapat diparsing langsung. Menggunakan sinyal simulasi untuk visualisasi pipeline.
                </div>
                """, unsafe_allow_html=True)
                t_raw, signal_raw, sr_val = generate_raw_audio(sentence_data, sr=sample_rate)
                signal_raw += np.random.normal(0, noise_level, len(signal_raw))
                run_full_pipeline(
                    sentence_data=sentence_data,
                    t_audio=t_raw, signal_audio=signal_raw, actual_sr=sr_val,
                    noise_level=noise_level, target_sr=sample_rate, n_mfcc=n_mfcc,
                    source_label="mic", stt_confidence=stt_conf
                )
        else:
            st.error("❌ Tidak dapat mengenali suara. Pastikan Anda berbicara dengan jelas dan coba rekam ulang.")
            st.markdown("""
            <div class="info-box">
                💡 <strong>Tips untuk hasil lebih baik:</strong><br>
                • Bicara dengan jelas dan tidak terlalu cepat<br>
                • Minimalisir suara latar belakang<br>
                • Gunakan mikrofon yang dekat dengan mulut<br>
                • Pastikan durasi rekaman cukup (minimal 1-2 detik)
            </div>
            """, unsafe_allow_html=True)

    else:
        # Belum ada rekaman — tampilan panduan
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="mode-tab">
                <h4>🎤 Siap Merekam</h4>
                <p>Klik ikon mikrofon di <strong>sidebar kiri</strong> untuk mulai merekam suara Anda.
                Klik sekali untuk mulai, klik lagi untuk berhenti.</p>
                <br>
                <p style="font-size:0.78rem; color:#64748b;">
                    Rekaman akan otomatis diproses melalui <strong>Google Speech API</strong>
                    untuk konversi suara → teks, kemudian divisualisasikan melalui seluruh pipeline STT.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------
# MODE 3: UPLOAD FILE AUDIO
# ---------------------------------------------------------------
elif input_mode == "📁 Upload File Audio":

    st.markdown("""
    <div class="info-box">
        📁 <strong>Mode: Upload File Audio</strong> — Pilih file audio di <strong>sidebar kiri</strong>
        (WAV, MP3, OGG, FLAC, M4A). File akan dikenali menggunakan Google Speech API,
        lalu divisualisasikan melalui pipeline STT lengkap.<br>
        💡 Untuk hasil terbaik, gunakan file <strong>WAV mono</strong> dengan durasi <strong>&lt; 30 detik</strong>.
    </div>
    """, unsafe_allow_html=True)

    if 'uploaded_file' in dir() and uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        file_bytes = uploaded_file.read()
        file_size_kb = len(file_bytes) / 1024

        st.markdown(f"""
        <div class="warn-box">
            ✅ <strong>File diterima:</strong> {uploaded_file.name}
            ({file_size_kb:.0f} KB, format: .{file_ext})
        </div>
        """, unsafe_allow_html=True)

        # Playback audio
        mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
                    "flac": "audio/flac", "m4a": "audio/mp4"}
        st.audio(file_bytes, format=mime_map.get(file_ext, "audio/wav"))

        # Konversi ke WAV jika bukan WAV
        wav_bytes = convert_audio_to_wav(file_bytes, file_ext)

        if wav_bytes is None:
            st.markdown("""
            <div class="warn-box">
                ⚠️ Format non-WAV memerlukan library <code>pydub</code> dan <code>ffmpeg</code> untuk konversi.
                Install dengan:<br>
                <code>pip install pydub</code> + install <code>ffmpeg</code> pada sistem.<br><br>
                Untuk saat ini, silakan upload file <strong>.wav</strong> langsung.
            </div>
            """, unsafe_allow_html=True)
        else:
            # --- Speech Recognition ---
            with st.spinner("🧠 Menjalankan Speech Recognition (Google API)..."):
                recognized_text, stt_conf = recognize_speech(wav_bytes, language=stt_language)

            if recognized_text:
                st.markdown(f"""
                <div class="result-box" style="margin-bottom:16px;">
                    🗣️ Teks terdeteksi: "{recognized_text}"
                </div>
                """, unsafe_allow_html=True)

                # Parse WAV ke sinyal untuk visualisasi
                t_audio, signal_audio, audio_sr = read_audio_bytes(wav_bytes)
                sentence_data = text_to_sentence_data(
                    recognized_text,
                    duration=(t_audio[-1] if t_audio is not None and len(t_audio) > 0 else None)
                )

                if t_audio is not None and sentence_data:
                    run_full_pipeline(
                        sentence_data=sentence_data,
                        t_audio=t_audio, signal_audio=signal_audio, actual_sr=audio_sr,
                        noise_level=noise_level, target_sr=sample_rate, n_mfcc=n_mfcc,
                        source_label="upload", stt_confidence=stt_conf
                    )
                elif sentence_data:
                    st.markdown("""
                    <div class="warn-box">
                        ⚠️ Sinyal audio tidak dapat diparsing langsung. Menggunakan sinyal simulasi untuk visualisasi.
                    </div>
                    """, unsafe_allow_html=True)
                    t_raw, signal_raw, sr_val = generate_raw_audio(sentence_data, sr=sample_rate)
                    signal_raw += np.random.normal(0, noise_level, len(signal_raw))
                    run_full_pipeline(
                        sentence_data=sentence_data,
                        t_audio=t_raw, signal_audio=signal_raw, actual_sr=sr_val,
                        noise_level=noise_level, target_sr=sample_rate, n_mfcc=n_mfcc,
                        source_label="upload", stt_confidence=stt_conf
                    )
            else:
                st.error("❌ Tidak dapat mengenali suara dalam file. Pastikan file berisi ucapan yang jelas.")
                st.markdown("""
                <div class="info-box">
                    💡 <strong>Tips untuk file audio:</strong><br>
                    • Gunakan format WAV mono, 16-bit, 16kHz<br>
                    • Durasi optimal: 2-30 detik<br>
                    • Pastikan berisi ucapan yang jelas<br>
                    • Minimalisir noise latar belakang
                </div>
                """, unsafe_allow_html=True)

    else:
        # Belum ada file — tampilan panduan
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="mode-tab">
                <h4>📁 Upload File Audio</h4>
                <p>Gunakan uploader di <strong>sidebar kiri</strong> untuk memilih file audio dari perangkat Anda.</p>
                <br>
                <p style="font-size:0.78rem; color:#64748b;">
                    <strong>Format didukung:</strong> WAV, MP3, OGG, FLAC, M4A<br>
                    <strong>Rekomendasi:</strong> WAV mono, 16kHz, durasi &lt; 30 detik<br>
                    <strong>Proses:</strong> File → Google Speech API → Pipeline Visualisasi lengkap
                </p>
            </div>
            """, unsafe_allow_html=True)


# =================================================================
# FOOTER: PENJELASAN TAHAPAN (selalu tampil di semua mode)
# =================================================================
st.markdown("---")
st.markdown("""
<div class="stage-card">
    <div class="stage-header">
        <div class="stage-num">📖</div>
        <div class="stage-title">Penjelasan Setiap Tahap Speech to Text</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **🎙️ Voice Capture (Input Audio)**

    Mikrofon menangkap getaran udara (gelombang suara) dan mengubahnya menjadi sinyal listrik.
    Komputer melakukan *sampling* — mengambil ribuan titik data per detik (sample rate)
    untuk merepresentasikan suara secara digital.

    **📊 Preprocessing (Pra-Pemrosesan)**

    Sinyal mentah dibersihkan dari noise (suara latar), dinormalisasi volumenya,
    dan dipotong bagian hening. Tahap ini memastikan data yang masuk ke model AI bersih dan konsisten.
    """)
with col2:
    st.markdown("""
    **🔬 Feature Extraction (Ekstraksi Fitur)**

    Sinyal diubah menjadi representasi matematis (MFCC — Mel-Frequency Cepstral Coefficients)
    yang menangkap karakteristik penting suara: nada, ritme, dan pola frekuensi.

    **🧠 Decoding (Inferensi Model)**

    Model AI (deep learning) membandingkan fitur akustik dengan pola bahasa yang telah
    dipelajari dari jutaan data latih, lalu memprediksi urutan kata yang paling mungkin.
    """)
