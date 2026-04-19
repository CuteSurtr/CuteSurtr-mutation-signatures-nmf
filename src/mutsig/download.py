from __future__ import annotations
import gzip
import io
import json
import time
from pathlib import Path
import pandas as pd
import requests
GDC_FILES = 'https://api.gdc.cancer.gov/files'
GDC_DATA = 'https://api.gdc.cancer.gov/data/'
COSMIC_URL = 'https://raw.githubusercontent.com/SigProfilerSuite/SigProfilerAssignment/main/SigProfilerAssignment/data/Reference_Signatures/GRCh38/COSMIC_v3.3_SBS_GRCh38.txt'
HEADERS = {'User-Agent': 'mutation-signatures-nmf/0.1'}
REQUEST_DELAY = 0.3

def query_gdc_maf_files(project_id: str, size: int) -> list[dict]:
    payload = {'filters': {'op': 'and', 'content': [{'op': 'in', 'content': {'field': 'cases.project.project_id', 'value': [project_id]}}, {'op': 'in', 'content': {'field': 'data_type', 'value': ['Masked Somatic Mutation']}}, {'op': 'in', 'content': {'field': 'access', 'value': ['open']}}]}, 'fields': 'file_id,file_name,file_size,cases.submitter_id,experimental_strategy,data_format', 'size': str(size), 'format': 'JSON'}
    r = requests.post(GDC_FILES, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()['data']['hits']

def download_maf(file_id: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    url = GDC_DATA + file_id
    r = requests.get(url, headers=HEADERS, stream=True, timeout=120)
    if r.status_code != 200:
        print(f'  ! GDC HTTP {r.status_code} for {file_id}')
        return False
    out_path.write_bytes(r.content)
    time.sleep(REQUEST_DELAY)
    return True

def fetch_project_mafs(project_id: str, out_dir: Path, limit: int=50) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f'{project_id}_manifest.json'
    if meta_path.exists():
        hits = json.loads(meta_path.read_text(encoding='utf-8'))
    else:
        hits = query_gdc_maf_files(project_id, size=limit)
        meta_path.write_text(json.dumps(hits, indent=2), encoding='utf-8')
    print(f'[{project_id}] found {len(hits)} open MAF files; downloading {min(limit, len(hits))}')
    paths: list[Path] = []
    for hit in hits[:limit]:
        fname = hit['file_name']
        local = out_dir / fname
        if download_maf(hit['file_id'], local):
            paths.append(local)
            if len(paths) % 10 == 0:
                print(f'  ... {len(paths)} files downloaded')
    return paths
MAF_COLUMNS = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode', 'CONTEXT']

def _read_one_maf(path: Path) -> pd.DataFrame:
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith('#'):
                f.seek(pos)
                break
        df = pd.read_csv(f, sep='\t', low_memory=False, usecols=lambda c: c in MAF_COLUMNS)
    return df

def load_project_mafs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            frames.append(_read_one_maf(p))
        except Exception as e:
            print(f'  ! failed to parse {p.name}: {e}')
    df = pd.concat(frames, ignore_index=True)
    df = df[df['Variant_Type'] == 'SNP'].copy()
    df = df.dropna(subset=['CONTEXT', 'Reference_Allele', 'Tumor_Seq_Allele2'])
    return df

def fetch_cosmic_sbs(out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path
    r = requests.get(COSMIC_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path
