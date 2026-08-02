# 🌌 Hand Gesture Hologram VFX

Aplikasi pengolahan citra real-time berbasis **OpenCV** dan **MediaPipe** yang menghasilkan efek visual hologram futuristik secara interaktif menggunakan gestur tangan dan webcam.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

---

## ✨ Fitur Utama

- ✋ **Hand Tracking & Gesture Recognition**: Deteksi 21 titik sendi tangan secara real-time dan stabil menggunakan **MediaPipe Hands**.
- 📈 **EMA Smoothing Filter**: Menghilangkan getaran (*jitter*) pada koordinat deteksi tangan dengan *Exponential Moving Average*.
- 🎨 **3 Varian Efek Hologram**:
  1. **`hologram.py` (Glitch & Forcefield Mesh)**: Efek *Chromatic Aberration*, kisi *Synthwave* diagonal, *laser scanline*, dan *glitch digital*.
  2. **`hologram_matrix.py` (Matrix Digital Rain)**: Efek hujan kode biner retro berwarna hijau khas *The Matrix*.
  3. **`hologram_vfx.py` (Neon Sci-Fi Wire-Frame)**: Efek garis deteksi tepi (*edge detection*) neon *cyan* bercahaya (*glow*).
- 📸 **Tangkapan Layar (Screenshot Laptop)**: Timer hitung mundur 4 detik untuk mengambil foto layar.
- 📹 **Perekam Layar (Screen Recorder Laptop)**: Perekaman layar desktop dengan timer hitung mundur 4 detik.

---

## 🖐️ Panduan Gestur Tangan

| Gestur | Aksi Visual |
|---|---|
| **Membentuk Bingkai** *(Jempol & Telunjuk 2 Tangan)* | Memunculkan hologram terisolasi di dalam area segi empat antara kedua tangan |
| **Dua Telapak Tangan Terbuka** | Memicu mode hologram **Full Screen** di seluruh layar webcam |
| **Kepalan Tangan** *(Satu / Kedua Tangan)* | Menyembunyikan/menonaktifkan efek hologram (hanya menampilkan *hand skeleton*) |

---

## ⌨️ Kontrol Keyboard

- **`S`** : Memulai timer 4 detik untuk **mengambil Screenshot layar laptop** (File disimpan sebagai `screenshot_laptop_<timestamp>.png`).
- **`R`** : Memulai timer 4 detik untuk **mulai/berhenti merekam layar laptop** (File disimpan sebagai `rekaman_layar_<timestamp>.avi`).
- **`Q`** : Keluar dari aplikasi.

---

## 🛠️ Persyaratan Sistem & Instalasi

### 1. Prasyarat
- **Python 3.8** atau versi yang lebih baru
- Webcam/Kamera laptop yang berfungsi

### 2. Instalasi Dependensi
Buka terminal/command prompt dan jalankan perintah berikut untuk menginstal pustaka yang dibutuhkan:

```bash
pip install opencv-python mediapipe numpy pillow
```

---

## 🚀 Cara Menjalankan

Pilih salah satu varian efek hologram yang ingin Anda jalankan:

### 1. Hologram Glitch & Forcefield (Default)
```bash
python hologram.py
```

### 2. Hologram Matrix Digital Rain
```bash
python hologram_matrix.py
```

### 3. Hologram Neon Sci-Fi Wire-Frame
```bash
python hologram_vfx.py
```

---

## 📂 Struktur Proyek

```text
.
├── hologram.py          # Script utama: Hologram Synthwave Glitch & Mesh Forcefield
├── hologram_matrix.py   # Script varian: Hologram Retro Green Matrix Binary Rain
├── hologram_vfx.py      # Script varian: Hologram Neon Cyan Sci-Fi Wire-frame
└── README.md            # Dokumentasi proyek
```

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).
