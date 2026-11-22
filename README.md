# Apex | F1 Race Predictor

Dashboard Streamlit untuk melihat ranking LightGBM dan membandingkannya dengan hasil balapan.

## Menjalankan aplikasi

Dari folder proyek, gunakan Python 3.10 atau lebih baru untuk memasang dependensi dan membuka aplikasi.

```bash
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

Arsip ini menyertakan model LightGBM dan empat contoh akhir pekan dengan data simulasi. Aplikasi menampilkan label **Data simulasi** saat memakai contoh tersebut.

## Menggunakan data balapan

Letakkan dataset Anda di `data/processed/f1_dataset_processed.parquet` dan muat ulang halaman. Aplikasi akan membaca dataset tersebut secara otomatis.

Jika Anda sudah memiliki file `f1_sessions_*.parquet` di `data/raw/`, jalankan `python src/features.py` untuk membentuk dataset. Untuk mengambil file mentah, pasang FastF1 (`python -m pip install fastf1`) dan jalankan `python src/extract.py`. Ekstraksi memerlukan koneksi internet.

Untuk dataset lama, aplikasi melengkapi gap finis dari snapshot F1DB ketika Grand Prix, pemenang, dan posisi pembalap cocok. Kolom opsional `finish_gap_seconds` dan `finish_gap_laps` pada dataset Anda dipakai lebih dulu. Dataset demo memakai gap simulasi sendiri.

## Membaca hasil

- **Skor ranking model** menentukan urutan prediksi. Untuk dua pembalap, skor yang lebih tinggi berarti posisi prediksi yang lebih baik.
- **Kecocokan podium** menghitung pembalap yang sama di tiga besar model dan hasil balapan, terlepas dari urutannya.
- **Sirkuit & gap** menempatkan titik pembalap di lintasan berdasarkan gap finis dari pemenang. Klik pembalap pada daftar gap untuk menyorot titiknya. Posisi titik diperkirakan dari gap waktu dan kecepatan rata-rata. Daftar menampilkan gap waktu yang tercatat. Jika tersedia, garis finis diperkirakan dari rute lap referensi. Peta lainnya memakai titik acuan P1. Pembalap yang tertinggal putaran dan status seperti DNF ditampilkan pada daftar status.
- **Lagu kebangsaan pemenang** berulang di latar belakang. Browser yang membatasi autoplay akan memulai audio setelah Anda berinteraksi dengan halaman. Gunakan tombol **Jeda** atau **Putar** untuk mengatur audio.
- **Buat prediksi** memakai Grand Prix terpilih sebagai dasar. Ubah grid, delta kualifikasi, delta pace FP2, serta perkiraan hujan dan suhu aspal. Tekan **Jalankan prediksi** untuk melihat podium dan perubahan posisi menurut model LightGBM.

## Sumber aset dan data

- Banner memakai gambar yang diunggah pengguna.
- Konfigurasi sirkuit, profil pembalap, dan catatan gap historis berasal dari [F1DB v2026.21.0](https://github.com/f1db/f1db/releases/tag/v2026.21.0) oleh Marcel Overdijk dan para kontributor. Data tabel dipilih dan diringkas dari musim 2018–2026. Rata-rata waktu putaran dihitung dari waktu finis pemenang dibagi jumlah putaran. Peta menggunakan jalur SVG dari rilis yang sama. Lisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) disertakan di `data/reference/LICENSE-F1DB.txt`.
- Perkiraan garis finis di `data/reference/track_timing.json` dihitung dengan mencocokkan bentuk peta dengan [putaran referensi dari f1data](https://github.com/martintxortiz/f1data/blob/main/docs/DATA_FORMAT.md).
- Rekaman United States Navy Band bersumber dari Wikimedia Commons dan dikonversi menjadi MP3. Judul serta sumber setiap rekaman ada di `data/reference/anthems.json`.
