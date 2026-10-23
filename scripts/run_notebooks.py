"""Execute the analysis notebooks in order with this Python interpreter.

Run from the repository root: python scripts/run_notebooks.py
Each notebook starts in a fresh kernel; outputs are saved in place.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [ROOT / 'notebooks' / name for name in (
    '01_clustering.ipynb', '02_classification.ipynb')]


def main() -> None:
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    with tempfile.TemporaryDirectory(prefix='bank_transactions_kernel_') as temporary:
        data_root = Path(temporary)
        spec = data_root / 'kernels' / 'bank-transactions'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Bank Transactions', 'language': 'python',
        }), encoding='utf-8')
        previous_path = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = str(data_root) + (os.pathsep + previous_path if previous_path else '')
        try:
            for path in NOTEBOOKS:
                print(f'Running {path.relative_to(ROOT)}', flush=True)
                notebook = nbformat.read(path, as_version=4)
                for cell in notebook.cells:
                    if cell.cell_type == 'code':
                        cell.outputs = []
                        cell.execution_count = None
                try:
                    NotebookClient(
                        notebook, timeout=1200, kernel_name='bank-transactions',
                        resources={'metadata': {'path': str(ROOT)}}, allow_errors=False,
                    ).execute()
                finally:
                    nbformat.write(notebook, path)
                print('Completed with saved outputs.', flush=True)
        finally:
            if previous_path is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = previous_path
    print('Both notebooks completed successfully.')


if __name__ == '__main__':
    main()
