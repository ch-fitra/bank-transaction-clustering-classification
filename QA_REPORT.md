# Audit akhir BMLP ? 23 Oktober 2026

ZIP final: `submission/BMLP_Chairul_Fitra_Ramadhan.zip` (2.227.879 byte).
Status: validasi lokal selesai; siap diunggah sendiri oleh pengguna. Nilai akhir tetap keputusan reviewer Dicoding.

| Kriteria | Hasil audit Advanced | Bukti (indeks cell mulai 0) |
|---|---|---|
| K1 EDA | Terpenuhi | Clustering 8/10/12: head/info/describe; 15: corr heatmap; 18: kelima numerik; 22: empat kategori dan tujuh ID/alamat/tanggal |
| K2 Preprocessing | Terpenuhi | 24?53: null, duplikat, dropna/drop_duplicates, filter ID/IP/date, LabelEncoder, IQR drop, StandardScaler, binning usia |
| K3 Clustering | Terpenuhi | 54?67: Yellowbrick asli, distortion + silhouette, KMeans, model joblib, PCA dua komponen + KMeans terpisah |
| K4 Interpretasi | Terpenuhi | 70?82: mean/min/max, interpretasi kedua cluster, inverse numerik/kategori, mode, ekspor kedua CSV |
| K5 Klasifikasi | Terpenuhi | Klasifikasi 4?21: inverse CSV, one-hot, split stratified, DT/RF, test metrics, train-only GridSearchCV, tiga model joblib |

Environment: Python 3.11.9 dalam `.venv`, scikit-learn 1.7.0, Yellowbrick 1.5 asli, pandas 2.3.3, NumPy 2.4.6, matplotlib 3.11.2. `pip check` lolos; versi lengkap tersimpan di `requirements.lock.txt`. Interpreter VS Code diarahkan ke `.venv`; kernel `Python 3.11 (.venv BMLP)` terdaftar. Eksekusi terverifikasi menggunakan interpreter `.venv` melalui kernelspec temporer tersendiri.

Dataset lokal resmi: 2.537 ? 16; 21 duplikat sebelum cleaning. URL publik dari template aktif. Kedua kolom tanggal pada URL memiliki nama saling tertukar dibanding CSV lokal; semua nilai lainnya identik setelah parsing numerik. Nilai kedua tanggal identik ketika dipertukarkan dan diparsing. Pertukaran tidak mengubah dropna/drop_duplicates ataupun fitur akhir karena kedua tanggal dibuang. Notebook memakai URL agar dapat diulang tanpa CSV mentah di ZIP; koneksi internet diperlukan.

Data akhir: 1.945 ? 11, yaitu 10 fitur + Target. Tidak ada null, kolom indeks CSV tambahan, atau ID/IP/tanggal. Target kedua CSV identik per baris. Round-trip StandardScaler/LabelEncoder lolos.

K utama = 2, silhouette = 0,572159925838334. Distortion elbow K=4; K=2 dipilih lewat silhouette maksimum dibanding K=2..9 (K=4 sekitar 0,445118), dengan alasan tertulis pada interpretasi. Cluster 0: 980 transaksi, Albuquerque?Louisville, rata-rata nilai 255,55 dan saldo 5.142,17. Cluster 1: 965 transaksi, Memphis?Washington, rata-rata nilai 258,15 dan saldo 5.058,81. Kuantil usia memiliki batas 18/32/55/80 tahun. LoginAttempts konstan 1 setelah IQR; keterbatasan seleksi ini dijelaskan.

Train = 1.556; test = 389 (kelas 0: 196, kelas 1: 193), indeks tidak overlap.

| Model | Accuracy test | Precision weighted | Recall weighted | F1 weighted |
|---|---:|---:|---:|---:|
| Decision Tree | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Random Forest | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Tuned Decision Tree | 1,0000 | 1,0000 | 1,0000 | 1,0000 |

GridSearchCV: 5 fold pada TRAIN, scoring f1_weighted. Best CV 0,9961413993122322; max_depth=None, min_samples_leaf=4, min_samples_split=2. Tuning tidak menaikkan skor test. Multiprocessing CV dinonaktifkan untuk menghindari cleanup resource tracker joblib di Windows; run final selesai tanpa traceback joblib.

Batasan: Target adalah pseudo-label KMeans; tidak tersedia ground-truth fraud. Semua lokasi pada data akhir hanya memiliki satu Target, dan Location menyumbang feature importance DT hampir 100%, RF sekitar 92,50%. LabelEncoder memaksakan jarak ordinal artifisial, sehingga silhouette dan akurasi tinggi tidak menunjukkan keberhasilan mendeteksi penipuan. Preprocessing, vocabulary kategori, dan clustering memakai keseluruhan dataset sebelum split classifier; evaluasi test merupakan reproduksi partisi tersebut, bukan estimasi prospektif pipeline fraud independen.

Verifikasi akhir:
- Dua notebook dijalankan berurutan dari kernel bersih; 35 dan 13 code cell dieksekusi berurutan, tidak ada error output atau placeholder kode.
- Struktur 84/23 cell, markdown template selain dua cell interpretasi, dan imports default dipertahankan.
- Delapan grafik tersimpan; heatmap, histogram, boxplot, countplot, ringkasan ID/tanggal, dua grafik pemilihan K, dan scatter diperiksa visual. Legenda scatter berada di luar area titik.
- Lima model buatan run terbaru dapat diload joblib; prediksi KMeans cocok dengan Target CSV; model PCA memprediksi dua cluster; prediksi ketiga classifier pada test diverifikasi ulang.
- Hash artefak dan waktu pembuatan model diverifikasi terhadap manifest eksekusi.
- `scripts/validate_and_package.py` lolos; ZIP dibuka ulang, CRC lolos, sembilan anggota tepat di root, berkas hasil ekstraksi cocok byte per byte, notebook/CSV/model hasil ekstraksi valid.
- Folder submission tepat berisi sembilan artefak dan satu ZIP; tidak ada dataset mentah, skrip, dependency, laporan audit, atau berkas tambahan dalam ZIP.

SHA256 ZIP: `6f69e5f3b9f15f81a901ff81d27ca1bea891fbcb5cef7edc19e090663b74917e`.
