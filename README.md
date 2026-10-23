# Bank Transaction Clustering & Classification — Dicoding Submission

Branch ini menyimpan versi submission Dicoding sebelum perapian proyek menjadi portofolio pribadi. Versi portofolio berada di [branch main](https://github.com/ch-fitra/bank-transaction-clustering-classification/tree/main).

## Berkas final

Gunakan `submission/BMLP_Chairul_Fitra_Ramadhan.zip` untuk submission. ZIP berisi tepat sembilan berkas: dua notebook dengan output, dua CSV hasil clustering, dan lima model joblib. Ekstensi model `.h5` mengikuti penamaan submission; format serialisasinya tetap joblib.

Notebook kerja di root mempertahankan struktur template asli. Dataset input adalah `bank_transactions_data_edited.csv`; template referensi berada di `templates/`. `QA_REPORT.md` memuat audit hasil dan batasan metode. Prompt dan instruksi untuk asisten AI tidak disertakan dalam versi branch ini.

## Menjalankan ulang di Windows

Gunakan Python 3.11 dan jalankan dari root proyek:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts/run_notebooks.py
.\.venv\Scripts\python.exe scripts/validate_and_package.py
```

Skrip menjalankan Clustering terlebih dahulu, kemudian Klasifikasi dari kernel bersih menggunakan interpreter yang menjalankannya. Setelah notebook selesai, validator memeriksa artefak dan membuat ulang ZIP. Notebook clustering menggunakan URL publik dataset yang tercantum pada template sehingga memerlukan koneksi internet.

Environment hasil terverifikasi: Python 3.11.9, scikit-learn 1.7.0, dan Yellowbrick 1.5 asli. Model dijalankan pada CPU. `.venv`, cache, model root hasil run, dan manifest eksekusi lokal dikecualikan dari Git; ZIP final sudah mencakup kelima model.

## Hasil dan batasan

- Data akhir: 1.945 transaksi, 10 fitur dan satu kolom Target.
- K-Means: dua cluster beranggotakan 980 dan 965 transaksi; silhouette 0,572160.
- Decision Tree, Random Forest, dan tuned Decision Tree: accuracy, precision weighted, recall weighted, dan F1 weighted sebesar 1,0000 pada 389 sampel test.
- CV tuning dilakukan pada train saja; best F1 weighted 0,996141. Tuning tidak meningkatkan skor test.

Target merupakan pseudo-label K-Means, bukan label fraud asli. Hasil clustering terutama dipengaruhi kode ordinal Location; preprocessing dan pembentukan cluster memakai keseluruhan data sebelum split classifier. Skor klasifikasi mengukur reproduksi partisi tersebut, bukan bukti keberhasilan mendeteksi penipuan.
