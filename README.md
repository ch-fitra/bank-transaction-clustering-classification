# Bank Transaction Clustering & Classification — Dicoding BMLP

Proyek submission Dicoding dengan dataset resmi transaksi bank: K-Means clustering menghasilkan **pseudo-label `Target`**, kemudian Decision Tree dan Random Forest mengklasifikasikan pseudo-label tersebut. **Bukan model fraud detection tervalidasi**.

## Mulai di VS Code (Windows / PowerShell)

1. Ekstrak ZIP starter dan buka folder `Bank-Transaction-Clustering-Classification` pada VS Code.
2. Baca **`PROMPT_CODEX.md`** lalu gunakan isi file itu sebagai instruksi Codex, atau kirim pesan: `Baca AGENTS.md dan PROMPT_CODEX.md lalu eksekusi seluruh tugas sampai submission ZIP siap. Jangan berhenti di rencana.`
3. Pasang ekstensi Microsoft `Python` dan `Jupyter` pada VS Code.
4. Jalankan PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import sklearn, yellowbrick; print(sklearn.__version__)"
```

5. Pilih Jupyter kernel `.venv` ketika membuka `.ipynb`. Code notebook memakai URL publik Google Sheets yang sudah ada dalam template resmi; internet dibutuhkan kecuali codex membuat fallback yang benar dan portabel.
6. Sesudah Codex audit/perbaiki notebook, eksekusi keduanya secara berurutan dari root (bisa klik **Run All** di VS Code, atau gunakan skrip di bawah):

```powershell
.\.venv\Scripts\python.exe scripts\run_notebooks.py
.\.venv\Scripts\python.exe scripts\validate_and_package.py
```

7. Bila selesai dan hasil valid, **upload `submission/BMLP_Chairul_Fitra_Ramadhan.zip`** ke Dicoding. Tidak perlu upload seluruh folder starter.

## Struktur

- `PROMPT_CODEX.md`: instruksi utama sangat rinci untuk Codex.
- `AGENTS.md`: petunjuk permanen proyek yang otomatis dapat dibaca Codex.
- `*.ipynb` di root: notebook kerja yang sebelumnya sudah diisi; harus diaudit/rerun dalam versi dependency target.
- `templates/`: notebook resmi Dicoding tanpa modifikasi, sebagai perbandingan struktur.
- `bank_transactions_data_edited.csv`: sumber data resmi untuk audit dan fallback lokal.
- `scripts/run_notebooks.py`: mengeksekusi notebook Clustering lalu Klasifikasi dengan working directory root.
- `scripts/validate_and_package.py`: QA berkas dan membuat ZIP final.
- `submission/`: keluaran final berisi 9 berkas + ZIP.

Jangan mengandalkan GPU NVIDIA: algoritma scikit-learn yang diwajibkan berjalan di CPU.


## Hasil terverifikasi

Run final menggunakan Python 3.11.9 dalam `.venv`, scikit-learn 1.7.0 dan Yellowbrick 1.5 asli. Semua cell kedua notebook dieksekusi berurutan dari kernel bersih, lalu validator packaging lolos. ZIP final berada di `submission/BMLP_Chairul_Fitra_Ramadhan.zip`.

Lihat `QA_REPORT.md` untuk bukti tiap kriteria, metrik test, audit grafik, perbedaan penamaan tanggal antara CSV lokal dan URL resmi, serta batasan pseudo-label. Snapshot dependency ada di `requirements.lock.txt`; manifest hash dan waktu eksekusi ada di `scripts/last_execution.json`. Laporan dan tools ini tidak masuk ZIP.

Untuk mengulang environment persis: `.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt`. Untuk memilih kernel di VS Code tersedia `Python 3.11 (.venv BMLP)`; skrip eksekusi selalu mengunci interpreter ke Python yang menjalankannya.
