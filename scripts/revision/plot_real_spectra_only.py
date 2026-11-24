#!/usr/bin/env python3
"""Create real measured vs 11 reconstructed spectra from a held-out HSIwheat plot.
No retraining, no whole-dataset inference; needs torch, numpy, pandas, matplotlib.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/trinity/home/jizh/datasets/HSIwheat_40'))
parser.add_argument('--device',default='cpu')
a = parser.parse_args()
root=a.root
out=root/'revision_review2_diagnostics'
out.mkdir(parents=True,exist_ok=True)
experiment=root/'experiment2_direct_mlp_3to40'
ckpts=sorted(experiment.rglob('best_direct_mlp_3to40.pth'))
if len(ckpts)!=11: raise RuntimeError(f'Expected 11 checkpoints, found {len(ckpts)}')
split=pd.read_csv(experiment/'cache'/'reconstruction_plot_split.csv',dtype={'field':str,'plot':str})
split=split.loc[split.recon_split.eq('test'),['field','plot']].drop_duplicates()

def norm(s):
    s=str(s).strip()
    return s[:-2] if s.endswith('.0') else s

selected=None
for rec in split.sort_values(['field','plot']).itertuples(index=False):
    folder=root/str(rec.field)/'hsi40'
    matches=[p for p in folder.glob('*.npy') if norm(p.stem)==norm(rec.plot)]
    if len(matches)!=1:continue
    cube=np.load(matches[0],mmap_mode='r')
    if cube.ndim!=3 or cube.shape[-1]!=40:continue
    flat=cube.reshape(-1,40)
    valid_idx=np.flatnonzero(np.any(flat!=0,axis=1))
    if len(valid_idx)==0:continue
    # One observed pixel selected a priori based solely on measured cube.
    stride=max(1,len(valid_idx)//20000)
    candidates=valid_idx[::stride]
    means=flat[candidates].mean(axis=1)
    pix=int(candidates[np.argmin(np.abs(means-np.median(means)))])
    selected=(rec.field,rec.plot,matches[0],pix,np.asarray(flat[pix],dtype=np.float32))
    break
if selected is None:raise RuntimeError('No valid held-out plot cube found')
field,plot,source,pix,gt=selected
wl=np.load(root/'wavelengths_40.npy')
if len(wl)!=40:raise RuntimeError('Incorrect wavelengths_40.npy')
print(f'Plot {field}/{plot} file={source.name} flattened_pixel_index={pix}',flush=True)
plt.figure(figsize=(11.5,6.3))
plt.plot(wl,gt,color='black',linewidth=2.7,label='Measured HSI')
rows=[]
for ckpt in ckpts:
    try:c=torch.load(ckpt,map_location='cpu',weights_only=False)
    except TypeError:c=torch.load(ckpt,map_location='cpu')
    model=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,40)).to(a.device)
    state=c['model_state_dict']
    if all(k.startswith('net.') for k in state):state={k[4:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model.eval()
    idx=np.asarray(c['selected_band_indices_40'],dtype=int)
    im=np.asarray(c['input_mean'],dtype=np.float32)
    sd=np.maximum(np.asarray(c['input_std'],dtype=np.float32),1e-12)
    om=np.asarray(c['output_mean40'],dtype=np.float32)
    os=np.maximum(np.asarray(c['output_std40'],dtype=np.float32),1e-12)
    x=(gt[idx]-im)/sd
    with torch.no_grad():z=model(torch.from_numpy(x[None]).to(a.device)).cpu().numpy()[0]
    pred=z*os+om
    name=ckpt.parent.relative_to(experiment).as_posix()
    plt.plot(wl,pred,linewidth=1.1,alpha=.85,label=name[:39])
    for b in range(40):rows.append({'field':field,'plot':plot,'flattened_pixel_index':pix,'triplet':name,'band_index_40':b,'wavelength_nm':float(wl[b]),'measured':float(gt[b]),'reconstructed':float(pred[b])})
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral value (original cube scale)')
plt.title(f'One actual held-out pixel: {field}/{plot}')
plt.legend(fontsize=6.4,ncol=2,loc='best')
plt.tight_layout()
figure=out/'real_measured_vs_reconstructed_spectra.png'
plt.savefig(figure,dpi=220)
plt.close()
pd.DataFrame(rows).to_csv(out/'real_spectral_example_values.csv',index=False)
print('Saved:',figure)
print('Saved:',out/'real_spectral_example_values.csv')