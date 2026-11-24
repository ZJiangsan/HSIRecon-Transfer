#!/usr/bin/env python3
"""Post-hoc feature deviation diagnostics for HSIRecon-Transfer.

Read-only analysis. Uses the frozen original benchmark and cached 81-D features.
No training or model checkpoint loading. Requires numpy, pandas, matplotlib.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/trinity/home/jizh/datasets/HSIwheat_40')
ap.add_argument('--output', default=None)
a = ap.parse_args()
root = Path(a.root).expanduser().resolve()
out = Path(a.output).expanduser().resolve() if a.output else root / 'revision_review2_diagnostics'
out.mkdir(parents=True, exist_ok=True)
exp = root / 'experiment2_direct_mlp_3to40'
measured_path = exp / 'cache' / 'measured40_features_frozen.npy'
meta_path = root / 'original190_paper_reproduction' / 'subplot_metadata.csv'
split_path = root / 'original190_paper_reproduction' / 'split_assignments.csv'
wl_path = root / 'wavelengths_40.npy'
for p in (measured_path, meta_path, split_path, wl_path):
    if not p.is_file(): raise FileNotFoundError(p)
meta = pd.read_csv(meta_path)
split = pd.read_csv(split_path)
keys = ['field', 'plot', 'subplot_index']
for df, desc in ((meta,'metadata'), (split,'split')):
    miss = [k for k in keys if k not in df.columns]
    if miss: raise ValueError(f'{desc} missing {miss}')
if 'split' not in split: raise ValueError('Frozen split lacks split column')
meta = meta.merge(split[keys + ['split']], on=keys, how='left', validate='one_to_one')
if meta['split'].isna().any(): raise ValueError('Unassigned subplots in frozen split')
train_idx = np.where(meta['split'].astype(str).to_numpy()=='train')[0]
test_idx = np.where(meta['split'].astype(str).to_numpy()=='test')[0]
if len(train_idx)==0 or len(test_idx)==0: raise ValueError('Empty train or test split')
x0 = np.load(measured_path, mmap_mode='r')
if x0.shape != (len(meta),81): raise ValueError(f'Unexpected reference shape {x0.shape}, expected {(len(meta),81)}')
wl = np.load(wl_path).reshape(-1)
if wl.shape != (40,): raise ValueError('Expected 40 wavelengths')
files = sorted(exp.rglob('yield_features.npz'))
if len(files)!=11: raise ValueError(f'Expected exactly 11 reconstructed feature caches, found {len(files)}: {files}')
# Frozen genuine-HSI training reference feature mean/std: no reconstructed or test info used for scales.
train = np.asarray(x0[train_idx], dtype=np.float64)
mu = train.mean(axis=0)
sigma = train.std(axis=0, ddof=0)
constant = sigma <= 1e-12
sigma[constant] = 1.0
print('Reference rows:', len(meta), 'train:', len(train_idx), 'test:', len(test_idx), 'test plots:',meta.iloc[test_idx][['field','plot']].drop_duplicates().shape[0], flush=True)
print('Frozen training features with near-zero SD:', np.where(constant)[0].tolist(),flush=True)
ref_test = np.asarray(x0[test_idx],dtype=np.float64)
# Plot IDs preserved for plotting/diagnostic grouped summaries; no resampling here.
test_meta = meta.iloc[test_idx].reset_index(drop=True)
plot_key = test_meta['field'].astype(str)+'::'+test_meta['plot'].astype(str)
by_plot = plot_key.groupby(plot_key).indices
rows=[]; bandrows=[]; plotrows=[]; series=[]
for ii,p in enumerate(files,1):
    label = str(p.parent.relative_to(exp)).replace('\\','/')
    with np.load(p) as z:
        if 'reconstructed40' not in z.files: raise ValueError(f'{p} lacks reconstructed40')
        if z['reconstructed40'].shape != x0.shape: raise ValueError(f'Wrong shape {p}: {z["reconstructed40"].shape}')
        cur = np.asarray(z['reconstructed40'][test_idx],dtype=np.float64)
    if not np.isfinite(cur).all(): raise ValueError(f'Nonfinite reconstruction features: {p}')
    area_maxerr = float(np.max(np.abs(cur[:,80]-ref_test[:,80])))
    if area_maxerr>1e-5: raise ValueError(f'Pixel-count alignment mismatch in {label}: max error {area_maxerr}')
    d = cur[:,:80]-ref_test[:,:80]
    dz = d / sigma[None,:80]
    sub={} 
    for group,start,end in [('mean',0,40),('std',40,80)]:
        e=d[:,start:end]; ez=dz[:,start:end]
        sub[group+'_rmse']=float(np.sqrt(np.mean(e**2)))
        sub[group+'_mae']=float(np.mean(np.abs(e)))
        sub[group+'_std_rmse']=float(np.sqrt(np.mean(ez**2)))
        sub[group+'_std_mae']=float(np.mean(np.abs(ez)))
        sub[group+'_bias']=float(np.mean(e))
        for b in range(40):
            ee=e[:,b]; zz=ez[:,b]
            bandrows.append({'representation':label,'feature_group':group,'band_index_40':b,
                             'wavelength_nm':float(wl[b]),'n_test_subplots':len(test_idx),
                             'feature_rmse':float(np.sqrt(np.mean(ee**2))),
                             'feature_mae':float(np.mean(np.abs(ee))),
                             'feature_bias':float(np.mean(ee)),
                             'train_standardized_rmse':float(np.sqrt(np.mean(zz**2))),
                             'train_standardized_mae':float(np.mean(np.abs(zz)))})
    sub['all80_train_standardized_rmse']=float(np.sqrt(np.mean(dz**2)))
    rows.append({'representation':label,'n_test_subplots':len(test_idx),'n_test_plots':len(by_plot),
                 'area_max_abs_error':area_maxerr,**sub})
    for group,idx in by_plot.items():
        ez=dz[idx]
        plotrows.append({'representation':label,'plot_key':group,'n_subplots':len(idx),
                         'mean_features_standardized_rmse':float(np.sqrt(np.mean(ez[:,:40]**2))),
                         'std_features_standardized_rmse':float(np.sqrt(np.mean(ez[:,40:]**2)))})
    print(f'[{ii}/{len(files)}] {label}: mean zRMSE={sub["mean_std_rmse"]:.4f} std zRMSE={sub["std_std_rmse"]:.4f}',flush=True)
summary = pd.DataFrame(rows)
bands=pd.DataFrame(bandrows)
plots=pd.DataFrame(plotrows)
summary.to_csv(out/'downstream_feature_deviation_summary.csv',index=False)
bands.to_csv(out/'downstream_feature_deviation_per_band.csv',index=False)
plots.to_csv(out/'downstream_feature_deviation_per_plot.csv',index=False)
# Optional descriptive comparison against original measured-HSI frozen model-transfer results.
transfer=out/'transfer_all_cases.csv'
if not transfer.exists():
    transfer=root/'prediction_model_transfer_model_classes'/'model_class_transfer_results_long.csv'
if transfer.exists():
    t=pd.read_csv(transfer)
    if {'representation','model_class','delta_R2'}.issubset(t.columns):
        t=t.loc[t['model_class'].astype(str).str.upper()=='MLP', ['representation','delta_R2']].copy()
        t['representation']=t['representation'].astype(str).str.replace('^direct_mlp::','',regex=True)
        merged=summary.merge(t, on='representation',how='left',validate='one_to_one')
        merged.to_csv(out/'downstream_feature_deviation_vs_transfer.csv',index=False)
        if merged['delta_R2'].notna().sum()>=3:
            for col in ('mean_std_rmse','std_std_rmse','all80_train_standardized_rmse'):
                corr=merged[[col,'delta_R2']].dropna().corr(method='spearman').iloc[0,1]
                print(f'DESCRIPTIVE Spearman({col}, MLP delta_R2)={corr:.4f}; 11 deliberately chosen conditions, not an inferential test',flush=True)
# Grouped feature deviation plot; common genuine-training standardized scale.
ix=np.arange(len(summary)); w=.38
fig,ax=plt.subplots(figsize=(12,5.5))
ax.bar(ix-w/2,summary['mean_std_rmse'],width=w,label='40 spectral means')
ax.bar(ix+w/2,summary['std_std_rmse'],width=w,label='40 spectral SDs')
ax.set_xticks(ix,summary['representation'],rotation=65,ha='right',fontsize=8)
ax.set_ylabel('RMSE in genuine-training SD units')
ax.set_title('Downstream 80 spectral feature deviations on held-out subplots')
ax.legend();fig.tight_layout();fig.savefig(out/'downstream_feature_deviation_by_triplet.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Per wavelength matrix for mean/SD feature deviations
fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
for ax,group in zip(axs,['mean','std']):
    v=bands[bands['feature_group']==group].pivot(index='representation',columns='band_index_40',values='train_standardized_rmse')
    v=v.reindex(summary['representation'])
    im=ax.imshow(v.to_numpy(),aspect='auto',interpolation='nearest')
    ax.set_yticks(np.arange(len(summary)),summary['representation'],fontsize=7)
    ax.set_title(('Mean-feature' if group=='mean' else 'SD-feature')+' standardized RMSE')
    fig.colorbar(im,ax=ax,shrink=.75,label='Training SD units')
axs[-1].set_xticks(np.arange(0,40,4),[f'{wl[i]:.0f}' for i in range(0,40,4)])
axs[-1].set_xlabel('Wavelength (nm)')
fig.savefig(out/'downstream_feature_deviation_heatmap.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Saved feature-deviation outputs to',out,flush=True)