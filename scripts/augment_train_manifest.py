#!/usr/bin/env python3
import argparse, json, subprocess, wave
from pathlib import Path

def duration_sec(path):
    with wave.open(str(path),'rb') as w: return w.getnframes()/w.getframerate()

def run(cmd): subprocess.run(cmd, check=True)
def safe_name(s): return ''.join(c if c.isalnum() else '_' for c in s.lower()).strip('_') or 'unknown'

def augment_audio(src, out_dir):
    src=Path(src); out_dir.mkdir(parents=True, exist_ok=True)
    configs=[
        ('speed095',['-filter:a','atempo=0.95']),
        ('speed105',['-filter:a','atempo=1.05']),
        ('volm3',['-filter:a','volume=-3dB']),
        ('volp3',['-filter:a','volume=3dB']),
        ('tel8k',['-ar','8000','-ac','1','-af','highpass=f=300,lowpass=f=3400']),
        ('soft_noise',['-filter_complex','[0:a]volume=0.95[a];anoisesrc=color=pink:amplitude=0.002[n];[a][n]amix=inputs=2:duration=first:dropout_transition=0']),
    ]
    outputs=[]
    for name, opts in configs:
        out=out_dir/f'{src.stem}_{name}.wav'; cmd=['ffmpeg','-y','-i',str(src)]+opts+['-ar','16000','-ac','1',str(out)]
        try: run(cmd); outputs.append((name,out.resolve()))
        except subprocess.CalledProcessError: print(f'[warn] augmentation failed: {name} {src}')
    return outputs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-manifest',required=True); ap.add_argument('--out-manifest',required=True); ap.add_argument('--out-audio-dir',default='data/audio_aug'); ap.add_argument('--keep-original',action='store_true'); args=ap.parse_args()
    rows=[json.loads(l) for l in Path(args.train_manifest).read_text(encoding='utf-8').splitlines() if l.strip()]
    final=[]
    for i,row in enumerate(rows,1):
        src=Path(row['audio_filepath'])
        if not src.exists(): print(f'[skip missing] {src}'); continue
        if args.keep_original:
            orig=dict(row); orig['augmentation']='original'; final.append(orig)
        out_dir=Path(args.out_audio_dir)/safe_name(row.get('use_case','unknown'))
        print(f'[{i}/{len(rows)}] augmenting {src}')
        for aug_name, aug_path in augment_audio(src,out_dir):
            new=dict(row); new['audio_filepath']=str(aug_path); new['duration']=round(duration_sec(aug_path),3); new['augmentation']=aug_name; final.append(new)
    out=Path(args.out_manifest); out.parent.mkdir(parents=True, exist_ok=True); out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in final)+'\n', encoding='utf-8')
    print('Original rows:',len(rows)); print('Final rows:',len(final)); print('Saved:',out)
if __name__=='__main__': main()
