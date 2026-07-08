#!/usr/bin/env python3
import argparse, json, random
from collections import defaultdict
from pathlib import Path

def read_manifest(path: Path):
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row=json.loads(line)
            if row.get('text','').strip(): rows.append(row)
    return rows

def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', default='data/manifests/aligned_chunk_manifest.json')
    ap.add_argument('--out-dir', default='data/manifests')
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    rows=read_manifest(Path(args.input)); groups=defaultdict(list)
    for r in rows: groups[r.get('use_case') or 'unknown'].append(r)
    keys=list(groups.keys()); random.seed(args.seed); random.shuffle(keys)
    if len(keys)<3: raise ValueError('Need at least 3 use cases/calls for train/val/test split.')
    train_n=max(1,int(len(keys)*0.70)); val_n=max(1,int(len(keys)*0.15))
    train_keys=set(keys[:train_n]); val_keys=set(keys[train_n:train_n+val_n]); test_keys=set(keys[train_n+val_n:])
    if not test_keys and val_keys:
        moved=sorted(val_keys)[-1]; val_keys.remove(moved); test_keys.add(moved)
    train=[r for k in train_keys for r in groups[k]]; val=[r for k in val_keys for r in groups[k]]; test=[r for k in test_keys for r in groups[k]]
    out=Path(args.out_dir)
    write_manifest(out/'train_aligned_manifest.json', train); write_manifest(out/'val_aligned_manifest.json', val); write_manifest(out/'test_aligned_manifest.json', test)
    print('========== SPLIT SUMMARY ==========')
    print('Total rows:',len(rows)); print('Total use cases:',len(keys)); print('Train rows:',len(train),sorted(train_keys)); print('Val rows:',len(val),sorted(val_keys)); print('Test rows:',len(test),sorted(test_keys))
if __name__=='__main__': main()
