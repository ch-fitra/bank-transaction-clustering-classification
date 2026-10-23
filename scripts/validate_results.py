"""Validate executed notebooks, generated datasets, and locally trained models.

Run after run_notebooks.py: python scripts/validate_results.py
Only load joblib files produced by these notebooks in your trusted workspace.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_notebooks() -> None:
    for name in ('01_clustering.ipynb', '02_classification.ipynb'):
        notebook = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
        nbformat.validate(notebook)
        code = [cell for cell in notebook.cells if cell.cell_type == 'code']
        require(bool(code), f'No code in {name}')
        require([cell.execution_count for cell in code] == list(range(1, len(code) + 1)),
                f'Incomplete or out-of-order execution in {name}')
        require(not any(output.output_type == 'error' for cell in code for output in cell.outputs),
                f'Error output in {name}')
        require(any(cell.outputs for cell in code), f'Missing outputs in {name}')
        if name.startswith('01'):
            images = sum('image/png' in output.get('data', {}) for cell in code for output in cell.outputs)
            require(images >= 5, 'Missing analysis figures')
        print(f'Validated notebook: {name} ({len(code)} code cells)')


def validate_results() -> dict:
    scaled = pd.read_csv(ARTIFACTS / 'clustered_transactions.csv')
    inverse = pd.read_csv(ARTIFACTS / 'clustered_transactions_inverse.csv')
    require(len(scaled) == len(inverse) and len(scaled) > 0, 'Dataset row mismatch')
    require(list(scaled.columns) == list(inverse.columns), 'Dataset schema mismatch')
    require(scaled.Target.equals(inverse.Target), 'Target order mismatch')
    require(scaled.Target.nunique() >= 2, 'Expected multiple clusters')
    for data in (scaled, inverse):
        require(not data.isna().any().any(), 'Missing data in exported CSV')
        require(not any(column.startswith('Unnamed:') for column in data), 'Unexpected CSV index')
        require(not any(token in column.lower() for column in data for token in ('id', 'ip', 'date')),
                'Identifier/address/date remains in features')
    require(len(inverse.select_dtypes('object').columns) > 0, 'Categories were not restored')

    features = scaled.drop(columns='Target')
    kmeans = joblib.load(ARTIFACTS / 'kmeans.joblib')
    check_is_fitted(kmeans)
    require(np.array_equal(kmeans.predict(features), scaled.Target), 'KMeans/CSV prediction mismatch')
    pca_model = joblib.load(ARTIFACTS / 'kmeans_pca.joblib')
    check_is_fitted(pca_model)
    projected = pd.DataFrame(PCA(n_components=2).fit_transform(features), columns=['PCA1', 'PCA2'])
    require(np.unique(pca_model.predict(projected)).size >= 2, 'PCA KMeans has one cluster')

    encoded = pd.get_dummies(inverse, columns=list(inverse.select_dtypes('object').columns), drop_first=True)
    X = encoded.drop(columns='Target')
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded.Target, test_size=0.2, random_state=42, stratify=encoded.Target)
    require(set(X_train.index).isdisjoint(X_test.index), 'Train/test index overlap')
    require('Target' not in X and 'Cluster' not in X, 'Label leakage in features')
    metrics = {}
    for name in ('decision_tree', 'random_forest', 'tuned_decision_tree'):
        model = joblib.load(ARTIFACTS / f'{name}.joblib')
        check_is_fitted(model)
        require(list(model.feature_names_in_) == list(X.columns), f'Feature schema mismatch: {name}')
        predicted = model.predict(X_test)
        metrics[name] = {
            'accuracy': accuracy_score(y_test, predicted),
            'precision_weighted': precision_score(y_test, predicted, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, predicted, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, predicted, average='weighted', zero_division=0),
        }
        if name == 'tuned_decision_tree':
            best_params = model.best_params_
            best_cv = float(model.best_score_)
    return {
        'rows': len(scaled), 'features': len(features.columns),
        'clusters': int(scaled.Target.nunique()),
        'cluster_counts': {str(key): int(value) for key, value in scaled.Target.value_counts().sort_index().items()},
        'silhouette': float(silhouette_score(features, scaled.Target)),
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'test_metrics': metrics, 'best_params': best_params, 'best_train_cv_f1_weighted': best_cv,
        'scope': 'Classification reproduces KMeans pseudo-labels; no ground-truth fraud evaluation.',
    }


def main() -> None:
    validate_notebooks()
    results = validate_results()
    (ARTIFACTS / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    print('Validation passed. Metrics saved to artifacts/metrics.json.')


if __name__ == '__main__':
    main()
