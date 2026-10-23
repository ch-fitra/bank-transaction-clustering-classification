"""Run the official Dicoding notebooks in order, saving real outputs.

Usage (from project root): python scripts/run_notebooks.py
This creates a temporary kernel spec pointing EXACTLY to this Python interpreter,
so a VS Code .venv won't accidentally execute with a global Jupyter kernel.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / '[Clustering]_Submission_Akhir_BMLP_Chairul_Fitra_Ramadhan.ipynb',
    ROOT / '[Klasifikasi]_Submission_Akhir_BMLP_Chairul_Fitra_Ramadhan.ipynb',
]


def main() -> int:
    started = datetime.now(timezone.utc)
    print(f'Working directory: {ROOT}')
    print(f'Kernel interpreter: {sys.executable}')
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix='bmlp_kernel_') as tmp:
        # JUPYTER_PATH points to a data root containing kernels/<name>.
        data_root = Path(tmp)
        spec_dir = data_root / 'kernels' / 'bmlp-current-env'
        spec_dir.mkdir(parents=True)
        (spec_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'BMLP Current Python Environment',
            'language': 'python',
        }), encoding='utf-8')
        os.environ['JUPYTER_PATH'] = str(data_root) + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        for path in SOURCES:
            if not path.exists():
                raise FileNotFoundError(path)
            print(f'\n>>> Running: {path.name}', flush=True)
            notebook = nbformat.read(path, as_version=4)
            try:
                client = NotebookClient(
                    notebook,
                    timeout=1200,
                    kernel_name='bmlp-current-env',
                    resources={'metadata': {'path': str(ROOT)}},
                    allow_errors=False,
                    on_cell_start=lambda cell, cell_index, **kwargs: print(
                        f'  cell {cell_index}', flush=True) if cell.cell_type == 'code' else None,
                )
                client.execute()
            except Exception:
                print(f'ERROR in {path.name}. Saving diagnostic outputs.', file=sys.stderr)
                nbformat.write(notebook, path)
                raise
            nbformat.write(notebook, path)
            count = sum(bool(c.get('outputs')) for c in notebook.cells if c.cell_type == 'code')
            print(f'OK: notebook executed, {count} cells have output.', flush=True)
    print('\nBoth notebooks executed successfully.')
    artifacts = SOURCES + list(ROOT.glob('*.h5')) + [ROOT / 'data_clustering.csv', ROOT / 'data_clustering_inverse.csv']
    manifest = {
        'started_utc': started.isoformat(),
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'executable': sys.executable,
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
    }
    (ROOT / 'scripts' / 'last_execution.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
