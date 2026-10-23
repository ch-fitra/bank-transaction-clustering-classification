# Instruksi permanen untuk Codex dalam proyek BMLP

Baca `PROMPT_CODEX.md` sepenuhnya sebelum memulai. Tujuan: dua notebook template Dicoding + hasil clustering dan klasifikasi, dengan target Advanced seluruh kriteria.

Aturan keras:
- Kerja di root proyek ini, Windows/VS Code, Python 3.11 + `.venv`, scikit-learn 1.7.0, Yellowbrick asli.
- Jangan mengganti template `.ipynb`, mengubah imports default, menambah cell tidak perlu, atau menggunakan AutoML.
- Dataset resmi `bank_transactions_data_edited.csv`, bukan dataset Kaggle pengganti.
- Audit notebook root yang sudah diisi, perbaiki seperlunya, rerun dari kernel bersih; jangan percaya output lama.
- Jalankan Clustering lebih dahulu, kemudian Classification; sertakan output notebook.
- Semua model di-dump joblib meskipun ekstensi `.h5`.
- Evaluasi klasifikasi wajib menggunakan test set, dan bedakan pseudo-label hasil cluster vs label fraud asli.
- Tidak ada pengiriman submission atas nama pengguna; hasil akhir adalah ZIP di `submission/`.
- Jangan mengklaim berhasil kalau gagal verifikasi; dokumentasikan blocker nyata.
