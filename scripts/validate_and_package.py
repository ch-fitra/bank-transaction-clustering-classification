"""Validate mandatory Dicoding BMLP submission files and build flat ZIP.

Run from root: python scripts/validate_and_package.py
Checks file presence, notebook errors/outputs, valid CSV targets, model load, archive CRC.
Does NOT replace a manual assessment of graphs, interpretability or rubric.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
import json
import hashlib
import io
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import nbformat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission'
STEM = 'BMLP_Chairul_Fitra_Ramadhan'
MODEL_FILES = [
    'model_clustering.h5',
    'PCA_model_clustering.h5',
    'decision_tree_model.h5',
    'explore_RandomForest_classification.h5',
    'tuning_classification.h5',
]
NOTEBOOK_FILES = [
    '[Clustering]_Submission_Akhir_BMLP_Chairul_Fitra_Ramadhan.ipynb',
    '[Klasifikasi]_Submission_Akhir_BMLP_Chairul_Fitra_Ramadhan.ipynb',
]
DATA_FILES = ['data_clustering.csv', 'data_clustering_inverse.csv']
EXPECTED = NOTEBOOK_FILES + MODEL_FILES + DATA_FILES


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def verify_notebooks() -> None:
    for name in NOTEBOOK_FILES:
        nb = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(nb)
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        require(code_cells, f'No code cells in {name}')
        incomplete = [i for i, c in enumerate(nb.cells)
                      if c.cell_type == 'code' and c.get('execution_count') is None]
        require(not incomplete, f'Unexecuted cells in {name}: {incomplete}')
        errors = [(i, out.get('ename', '?'), out.get('evalue', '?'))
                  for i, c in enumerate(nb.cells) if c.cell_type == 'code'
                  for out in c.get('outputs', []) if out.get('output_type') == 'error']
        require(not errors, f'Notebook errors in {name}: {errors}')
        num_outputs = sum(bool(c.get('outputs')) for c in code_cells)
        require(num_outputs >= (13 if 'Clustering' in name else 7),
                f'Notebook has insufficient outputs: {name} ({num_outputs})')
        text = '\n'.join(c.source for c in code_cells)
        if 'Clustering' in name:
            for marker in ('head()', 'info()', 'describe()', 'dropna(', 'drop_duplicates(',
                           'LabelEncoder(', 'StandardScaler(', 'KElbowVisualizer(',
                           'KMeans(', 'silhouette_score(', 'PCA(', 'inverse_transform('):
                require(marker in text, f'Missing key operation {marker} in Clustering')
            requires_images = 4
        else:
            for marker in ('train_test_split(', 'DecisionTreeClassifier(',
                           'RandomForestClassifier(', 'GridSearchCV(',
                           'classification_report(', 'X_test', 'y_test'):
                require(marker in text, f'Missing key operation {marker} in Classification')
            requires_images = 0
        images = sum(any('image/png' in out.get('data', {}) for out in c.get('outputs', []))
                     for c in code_cells)
        require(images >= requires_images,
                f'Insufficient plot outputs in {name}: {images}')
        print(f'VALID NOTEBOOK: {name}: code={len(code_cells)}, outputs={num_outputs}, plots={images}')


def verify_csv() -> None:
    basic = pd.read_csv(ROOT / DATA_FILES[0])
    inverse = pd.read_csv(ROOT / DATA_FILES[1])
    require(len(basic) >= 100, 'Too few rows after preprocessing')
    require(len(basic) == len(inverse), 'CSV row counts do not match')
    require('Target' in basic and 'Target' in inverse, 'Target column missing')
    require(basic['Target'].equals(inverse['Target']), 'Target ordering differs')
    require(basic['Target'].nunique() >= 2, 'At least 2 clusters required')
    require(not basic.isna().any().any() and not inverse.isna().any().any(), 'CSV has null values')
    require(not any(c.startswith('Unnamed:') for c in basic.columns), 'CSV exported with index')
    require(set(basic.columns) == set(inverse.columns), 'Inverse CSV feature mismatch')
    illegal = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID',
               'TransactionDate', 'PreviousTransactionDate']
    require(not (set(illegal) & set(basic.columns)), 'Identifier/Date columns remain')
    require(len(inverse.select_dtypes(include=['object']).columns) >= 1,
            'Inverse dataset must restore categorical columns')
    print(f'VALID CSV: {len(basic)} rows, {len(basic.columns)} columns; '
          f'Target classes={basic.Target.nunique()}, counts={basic.Target.value_counts().to_dict()}')


def verify_models() -> None:
    for name in MODEL_FILES:
        p = ROOT / name
        require(p.stat().st_size > 1000, f'Model suspiciously small: {name}')
        model = joblib.load(p)  # Only models produced by our own trusted notebooks.
        require(hasattr(model, 'predict'), f'Model cannot predict: {name}')
        if 'clustering' in name:
            require(hasattr(model, 'cluster_centers_'), f'Unfitted KMeans: {name}')
        else:
            require(hasattr(model, 'classes_') or hasattr(model, 'best_estimator_'),
                    f'Unfitted classifier: {name}')
        print(f'VALID MODEL: {name}, {type(model).__name__}')


def verify_execution() -> None:
    manifest = json.loads((ROOT / 'scripts' / 'last_execution.json').read_text(encoding='utf-8'))
    for name in EXPECTED:
        p = ROOT / name
        require(hashlib.sha256(p.read_bytes()).hexdigest() == manifest['sha256'][name],
                f'Artifact changed since clean execution: {name}')
        if name in MODEL_FILES:
            require(p.stat().st_mtime >= datetime.fromisoformat(manifest['started_utc']).timestamp(),
                    f'Stale model: {name}')
    for name in NOTEBOOK_FILES:
        nb = nbformat.read(ROOT / name, as_version=4)
        template = nbformat.read(ROOT / 'templates' / name.replace('Chairul_Fitra_Ramadhan', 'Your_Name'), as_version=4)
        require([c.cell_type for c in nb.cells] == [c.cell_type for c in template.cells], 'Template cell structure changed')
        import_index = 3 if 'Clustering' in name else 2
        require(nb.cells[import_index].source == template.cells[import_index].source, 'Default imports changed')
        for i, (c, t) in enumerate(zip(nb.cells, template.cells)):
            if c.cell_type == 'markdown' and i not in (71, 79):
                require(c.source == t.source, f'Template markdown changed: {name}, cell {i}')
        counts = [c.execution_count for c in nb.cells if c.cell_type == 'code']
        require(counts == list(range(1, len(counts) + 1)), f'Execution order invalid: {name}')
    print('VALID EXECUTION: clean sequential runs, artifact hashes/freshness, template structure/imports')


def verify_predictions() -> None:
    basic = pd.read_csv(ROOT / DATA_FILES[0])
    inverse = pd.read_csv(ROOT / DATA_FILES[1])
    features = basic.drop(columns='Target')
    cluster = joblib.load(ROOT / MODEL_FILES[0])
    require(np.array_equal(cluster.predict(features), basic.Target), 'KMeans labels do not reproduce CSV')
    projected = pd.DataFrame(PCA(n_components=2).fit_transform(features), columns=['PCA1', 'PCA2'])
    pca_labels = joblib.load(ROOT / MODEL_FILES[1]).predict(projected)
    require(len(np.unique(pca_labels)) >= 2, 'PCA KMeans predicts one cluster only')
    encoded = pd.get_dummies(inverse, columns=list(inverse.select_dtypes(include='object').columns), drop_first=True)
    X = encoded.drop(columns='Target')
    X_train, X_test, y_train, y_test = train_test_split(X, encoded.Target, test_size=0.2, random_state=42, stratify=encoded.Target)
    require(set(X_train.index).isdisjoint(X_test.index), 'Train/test overlap')
    for name in MODEL_FILES[2:]:
        model = joblib.load(ROOT / name)
        require(list(model.feature_names_in_) == list(X.columns), f'Classifier feature mismatch: {name}')
        pred = model.predict(X_test)
        print(f'TEST {name}: accuracy={accuracy_score(y_test, pred):.6f}, '
              f'precision={precision_score(y_test, pred, average="weighted", zero_division=0):.6f}, '
              f'recall={recall_score(y_test, pred, average="weighted", zero_division=0):.6f}, '
              f'F1={f1_score(y_test, pred, average="weighted", zero_division=0):.6f}')


def main() -> int:
    missing = [name for name in EXPECTED if not (ROOT / name).is_file()]
    require(not missing, f'Missing generated files in project root: {missing}. '
            'Run Clustering and Classification from the project root first.')
    verify_notebooks()
    verify_csv()
    verify_models()
    verify_execution()
    verify_predictions()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        shutil.copy2(ROOT / name, OUT / name)
    zip_path = OUT / f'{STEM}.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=7) as archive:
        for name in EXPECTED:
            archive.write(OUT / name, arcname=name)
    with zipfile.ZipFile(zip_path) as archive:
        require(sorted(archive.namelist()) == sorted(EXPECTED), 'ZIP contents differ from expected')
        require(archive.testzip() is None, 'ZIP CRC validation failed')
        require(all(archive.getinfo(n).file_size > 0 for n in EXPECTED), 'Empty item in ZIP')
        with tempfile.TemporaryDirectory(prefix='bmlp_zip_check_') as tmp:
            archive.extractall(tmp)
            for name in EXPECTED:
                require((Path(tmp) / name).read_bytes() == (ROOT / name).read_bytes(), f'ZIP content differs: {name}')
            for name in NOTEBOOK_FILES:
                nbformat.validate(nbformat.read(Path(tmp) / name, as_version=4))
            for name in DATA_FILES:
                require(pd.read_csv(Path(tmp) / name).equals(pd.read_csv(ROOT / name)), f'ZIP CSV invalid: {name}')
            for name in MODEL_FILES:
                require(hasattr(joblib.load(Path(tmp) / name), 'predict'), f'ZIP model invalid: {name}')
    require({p.name for p in OUT.iterdir()} == set(EXPECTED + [zip_path.name]), 'Submission folder has extra files')
    print('\nREADY: ' + str(zip_path))
    print(f'ZIP has {len(EXPECTED)} files; verified CRC and structure.')
    print('NOTE: Visual correctness, rubric alignment, and data interpretation still need human/agent review.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'VALIDATION FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
