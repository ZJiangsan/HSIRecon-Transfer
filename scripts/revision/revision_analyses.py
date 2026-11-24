#!/usr/bin/env python3
"""Reviewer-focused, NON-TRAINING diagnostics for HSIRecon-Transfer.

Original pipeline: https://github.com/ZJiangsan/HSIRecon-Transfer
Uses saved split/metadata, prediction CSVs and saved DirectMLP checkpoints.
No model training; no changes to original experiment outputs.

Examples:
 python revision_analyses.py audit --root /home/nibio/HSIwheat_40
 python revision_analyses.py bootstrap --root /home/nibio/HSIwheat_40 --nboot 5000
 python revision_analyses.py transfer-table --root /home/nibio/HSIwheat_40
 python revision_analyses.py spectra --root /home/nibio/HSIwheat_40 --max-models 11
"""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd


def plot_str(s):
    t = str(s).strip()
    return t[:-2] if t.endswith('.0') else t


def paths(root):
    exp = root / 'experiment2_direct_mlp_3to40'
    return {
        'exp': exp,
        'stage': root / 'original190_paper_reproduction',
        'measured': root / 'frozen_measured40_model_transfer',
        'classes': root / 'prediction_model_transfer_model_classes',
        'split': exp / 'cache' / 'reconstruction_plot_split.csv',
    }


def metadata(root):
    p=paths(root)
    m=pd.read_csv(p['stage']/'subplot_metadata.csv')
    s=pd.read_csv(p['stage']/'split_assignments.csv')
    keys=['field','plot','subplot_index']
    for a in (m,s):
        a['field']=a['field'].astype(str)
        a['plot']=a['plot'].map(plot_str)
        a['subplot_index']=a['subplot_index'].astype(str)
    if 'split' in m.columns:
        m=m.drop(columns=['split'])
    merged=m.merge(s[keys+['split']],on=keys,how='left',validate='one_to_one')
    assert merged['split'].notna().all(), 'Unmatched split rows'
    return merged.reset_index(drop=True)


def audit(root,out):
    p=paths(root)
    checks=[]
    for label,fn in [
        ('40 band wavelength vector',root/'wavelengths_40.npy'),
        ('original-190 band target indices',root/'target_indices_190.npy'),
        ('subplot metadata',p['stage']/'subplot_metadata.csv'),
        ('frozen subplot split',p['stage']/'split_assignments.csv'),
        ('reconstruction plot split',p['split']),
        ('primary transfer predictions',p['measured']/'frozen_measured40_primary_predictions.csv'),
        ('model class predictions',p['classes']/'model_class_transfer_predictions.csv'),
        ('model class results',p['classes']/'model_class_transfer_results_long.csv'),
    ]:
        checks.append({'item':label,'found':fn.exists(),'path':str(fn)})
    ckpts=sorted(p['exp'].rglob('best_direct_mlp_3to40.pth'))
    checks.append({'item':'DirectMLP reconstruction checkpoints','found':len(ckpts)==11,'path':f'{len(ckpts)} found'})
    for f in ckpts:
        checks.append({'item':'checkpoint','found':True,'path':str(f)})
    pd.DataFrame(checks).to_csv(out/'input_audit.csv',index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    if (p['stage']/'subplot_metadata.csv').exists() and (p['stage']/'split_assignments.csv').exists():
        m=metadata(root)
        test=m[m.split=='test']
        n=test[['field','plot']].drop_duplicates().shape[0]
        print(f'Frozen test set: {len(test)} subplots within {n} plots')
        print(test.groupby('field')[['plot']].nunique().to_string())
    return ckpts


def load_prediction_long(root):
    p=paths(root)
    file=p['classes']/'model_class_transfer_predictions.csv'
    if not file.exists():
        file=p['measured']/'frozen_measured40_primary_predictions.csv'
        if not file.exists():
            raise FileNotFoundError('No prediction CSV found. Run original scripts/04 and /05 first.')
        d=pd.read_csv(file)
        d['model_class']='MLP'
    else:
        d=pd.read_csv(file)
    needed=['model_class','representation','row_index','observed_yield_g','predicted_yield_g']
    assert all(c in d for c in needed),f'Missing columns {set(needed)-set(d.columns)}'
    meta=metadata(root)
    test_ids=set(meta.index[meta.split=='test'])
    d['row_index']=pd.to_numeric(d.row_index).astype(int)
    assert set(d.row_index)==test_ids, 'Saved predictions do not align with current frozen test split'
    map_keys=meta[['field','plot']].copy()
    d['field']=d.row_index.map(map_keys.field)
    d['plot']=d.row_index.map(map_keys['plot']).map(plot_str)
    d['plot_key']=d['field'].astype(str)+'/'+d['plot']
    if d.duplicated(['model_class','representation','row_index']).any():
        raise ValueError('Duplicate model/representation/test-row in saved predictions')
    return d


def metric(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    sse=np.sum((p-y)**2)
    sst=np.sum((y-y.mean())**2)
    return {'R2':1-sse/sst if sst>0 else np.nan,
            'RMSE_g':np.sqrt(np.mean((p-y)**2)), 'MAE_g':np.mean(np.abs(p-y))}


def transfer_table(root,out):
    d=load_prediction_long(root)
    rows=[]
    for (model,rep),g in d.groupby(['model_class','representation'],sort=True):
        x=metric(g.observed_yield_g,g.predicted_yield_g)
        rows.append({'model_class':model,'representation':rep,'N_subplots':len(g),**x})
    t=pd.DataFrame(rows)
    for model in t.model_class.unique():
        idx=t.model_class==model
        base=t[idx & t.representation.eq('measured40')]
        if len(base)!=1:raise ValueError(f'Baseline for {model}: {len(base)} rows')
        b=base.iloc[0]
        t.loc[idx,'delta_R2']=b.R2-t.loc[idx,'R2']
        t.loc[idx,'delta_RMSE_g']=t.loc[idx,'RMSE_g']-b.RMSE_g
        t.loc[idx,'delta_MAE_g']=t.loc[idx,'MAE_g']-b.MAE_g
        t.loc[idx,'RMSE_ratio_to_measured']=t.loc[idx,'RMSE_g']/b.RMSE_g
    t=t.sort_values(['model_class','representation'])
    t.to_csv(out/'transfer_all_cases.csv',index=False)
    print(t.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    return t


def bootstrap(root,out,nboot,seed):
    d=load_prediction_long(root)
    rng=np.random.default_rng(seed)
    reps=sorted(d.representation.unique())
    models=sorted(d.model_class.unique())
    b_rows=[];ci_rows=[]
    for model in models:
        sub=d[d.model_class==model]
        baseline=sub[sub.representation=='measured40'].set_index('row_index').sort_index()
        if baseline.empty:raise ValueError(f'Missing measured40 baseline for {model}')
        keys=baseline.plot_key.to_numpy(); groups=np.unique(keys); G=len(groups)
        if G!=50:raise ValueError(f'Expected exactly 50 held-out plots, found {G}')
        # Field-stratified bootstrap retains observed C3/C4/C9 plot counts.
        field=np.array([str(k).split('/')[0] for k in groups])
        bcounts=np.zeros((nboot,G),dtype=np.int16)
        for ff in np.unique(field):
            inds=np.flatnonzero(field==ff)
            random_select=rng.choice(inds,size=(nboot,len(inds)),replace=True)
            for i in range(nboot):bcounts[i]+=np.bincount(random_select[i],minlength=G).astype(np.int16)
        g_of_row=pd.Categorical(keys,categories=groups).codes
        y=baseline.observed_yield_g.to_numpy(dtype=float)
        n=np.bincount(g_of_row,minlength=G)
        sy=np.bincount(g_of_row,weights=y,minlength=G)
        sy2=np.bincount(g_of_row,weights=y*y,minlength=G)
        nboot_tot=bcounts@n
        y_tot=bcounts@sy
        sy2_tot=bcounts@sy2
        ss_total=sy2_tot-y_tot*y_tot/nboot_tot
        def summaries(pred):
            err=pred-y
            sse=bcounts@np.bincount(g_of_row,weights=err*err,minlength=G)
            sae=bcounts@np.bincount(g_of_row,weights=np.abs(err),minlength=G)
            return np.column_stack([1-sse/ss_total,np.sqrt(sse/nboot_tot),sae/nboot_tot])
        base_pred=baseline.predicted_yield_g.to_numpy(dtype=float)
        base_boot=summaries(base_pred)
        base_observed=metric(y,base_pred)
        for rep in reps:
            one=sub[sub.representation==rep].set_index('row_index').sort_index()
            if len(one)==0:continue
            if not one.index.equals(baseline.index):raise ValueError(f'Unaligned test rows {model} {rep}')
            if not np.allclose(one.observed_yield_g.to_numpy(dtype=float),y):raise ValueError('Labels differ')
            other=one.predicted_yield_g.to_numpy(dtype=float)
            bb=summaries(other)
            observed=metric(y,other)
            vals=np.column_stack([bb[:,0],bb[:,1],bb[:,2],base_boot[:,0]-bb[:,0],bb[:,1]-base_boot[:,1],bb[:,2]-base_boot[:,2]])
            point=np.array([observed['R2'],observed['RMSE_g'],observed['MAE_g'],base_observed['R2']-observed['R2'],observed['RMSE_g']-base_observed['RMSE_g'],observed['MAE_g']-base_observed['MAE_g']])
            names=['R2','RMSE_g','MAE_g','delta_R2','delta_RMSE_g','delta_MAE_g']
            for j,name in enumerate(names):
                lo,hi=np.quantile(vals[:,j],[0.025,0.975])
                ci_rows.append({'model_class':model,'representation':rep,'metric':name,'point_estimate':point[j], 'CI_95_low':lo,'CI_95_high':hi,'n_plots':G,'n_boot':nboot,'seed':seed,'bootstrap':'field-stratified plot cluster'})
            if rep!='measured40':
                b_rows.append({'model_class':model,'representation':rep,'delta_R2':point[3], 'delta_R2_CI95_low':np.quantile(vals[:,3],0.025),'delta_R2_CI95_high':np.quantile(vals[:,3],0.975)})
    ci=pd.DataFrame(ci_rows)
    ci.to_csv(out/'plot_cluster_bootstrap_all_metrics.csv',index=False)
    pd.DataFrame(b_rows).to_csv(out/'plot_cluster_bootstrap_summary.csv',index=False)
    print(pd.DataFrame(b_rows).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    return ci


def spectra(root,out,limit,device_name):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch import nn
    p=paths(root)
    ckpts=sorted(p['exp'].rglob('best_direct_mlp_3to40.pth'))
    if not ckpts:raise FileNotFoundError('No reconstruction checkpoints saved. Run original scripts 02 and 03.')
    if len(ckpts)!=11:print(f'WARNING: found {len(ckpts)} checkpoints, expected 11',file=sys.stderr)
    if limit>0:ckpts=ckpts[:limit]
    split=pd.read_csv(p['split'])
    for col in ['field','plot']:
        split[col]=split[col].astype(str)
    split['plot']=split['plot'].map(plot_str)
    tests=split[split.recon_split=='test'][['field','plot']].drop_duplicates()
    if len(tests)!=50:raise ValueError(f'Expected 50 held-out reconstruction plots, found {len(tests)}')
    wl=np.load(root/'wavelengths_40.npy')
    dev=torch.device(device_name)
    detail=[];examples={}
    sample_plot=tests.sort_values(['field','plot']).iloc[0]
    for ip,ckpt_file in enumerate(ckpts):
        print(f'[{ip+1}/{len(ckpts)}] {ckpt_file}',flush=True)
        try: c=torch.load(ckpt_file,map_location='cpu',weights_only=False)
        except TypeError:c=torch.load(ckpt_file,map_location='cpu')
        idx=np.asarray(c['selected_band_indices_40'],dtype=int)
        imean=np.asarray(c['input_mean'],dtype=np.float32)
        istd=np.asarray(c['input_std'],dtype=np.float32)
        omean=np.asarray(c['output_mean40'],dtype=np.float32)
        ostd=np.asarray(c['output_std40'],dtype=np.float32)
        istd=np.where(istd<1e-12,1,istd)
        ostd=np.where(ostd<1e-12,1,ostd)
        net=nn.Sequential(nn.Linear(3,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,40)).to(dev)
        # Original checkpoint was saved from DirectMLP3to40 with a .net submodule.
        # This diagnostic uses bare nn.Sequential: remove only the expected prefix.
        saved_state = c['model_state_dict']
        expected_keys = set(net.state_dict())
        if set(saved_state) == expected_keys:
            state = saved_state
        elif all(k.startswith('net.') for k in saved_state):
            state = {k.removeprefix('net.'): v for k, v in saved_state.items()}
        else:
            raise RuntimeError(f'Unexpected checkpoint keys in {ckpt_file}: {list(saved_state)[:8]}')
        if set(state) != expected_keys:
            raise RuntimeError(f'Checkpoint architecture mismatch in {ckpt_file}: {set(state) ^ expected_keys}')
        net.load_state_dict(state, strict=True)
        net.eval()
        sse=np.zeros(40);ae=np.zeros(40);bias=np.zeros(40);N=0
        example=None
        for record in tests.itertuples(index=False):
            file=root/record.field/'hsi40'/(str(record.plot)+'.npy')
            if not file.exists():
                alt=list((root/record.field/'hsi40').glob('*.npy'))
                matches=[z for z in alt if plot_str(z.stem)==record.plot]
                if len(matches)!=1:raise FileNotFoundError(f'Cube for {record.field}/{record.plot}: {len(matches)} matches')
                file=matches[0]
            cube=np.load(file,mmap_mode='r'); flat=np.asarray(cube.reshape(-1,40))
            valid=np.any(flat!=0,axis=1)
            if not np.any(valid):continue
            gt=flat[valid].astype(np.float32)
            if record.field==sample_plot.field and record.plot==sample_plot.plot:
                # Use the same deterministic pixel across triplets; closest to mean total reflectance.
                sample_i=int(np.argmin(np.abs(gt.mean(axis=1)-np.median(gt.mean(axis=1)))))
            for begin in range(0,len(gt),65536):
                raw=gt[begin:begin+65536]
                inp=(raw[:,idx]-imean)/istd
                with torch.no_grad():
                    pred=net(torch.from_numpy(inp).to(dev)).cpu().numpy()*ostd+omean
                error=pred.astype(np.float64)-raw
                sse+=np.square(error).sum(axis=0)
                ae+=np.abs(error).sum(axis=0)
                bias+=error.sum(axis=0)
                N+=len(raw)
                if record.field==sample_plot.field and record.plot==sample_plot.plot and begin<=sample_i<begin+len(raw):
                    example=(raw[sample_i-begin].copy(),pred[sample_i-begin].copy())
        tag=ckpt_file.parent.relative_to(p['exp']).as_posix().replace('/','__')
        for b,w in enumerate(wl):
            detail.append({'triplet':tag,'wavelength_nm':float(w),'band_index_40':b,'rmse':np.sqrt(sse[b]/N),'mae':ae[b]/N,'bias_pred_minus_measured':bias[b]/N,'n_test_pixels':N})
        if example is not None:examples[tag]=example
    df=pd.DataFrame(detail);df.to_csv(out/'per_wavelength_reconstruction_error.csv',index=False)
    plt.figure(figsize=(10,5.5))
    for k,g in df.groupby('triplet'):plt.plot(g.wavelength_nm,g.rmse,label=k[:38],linewidth=1.5)
    plt.xlabel('Wavelength (nm)');plt.ylabel('Per-band RMSE (original cube scale)')
    plt.legend(fontsize=6,ncol=2);plt.tight_layout();plt.savefig(out/'per_wavelength_rmse.png',dpi=230);plt.close()
    if examples:
        plt.figure(figsize=(10,5.5))
        first=next(iter(examples.values()))[0]
        plt.plot(wl,first,color='black',linewidth=2,label='Measured (same held-out pixel)')
        for k,(_gt,pred) in examples.items():plt.plot(wl,pred,alpha=.8,label=k[:38])
        plt.xlabel('Wavelength (nm)');plt.ylabel('Spectral value (original cube scale)')
        plt.title(f'Real test pixel: {sample_plot.field}/{sample_plot.plot}')
        plt.legend(fontsize=6,ncol=2);plt.tight_layout();plt.savefig(out/'real_measured_vs_reconstructed_spectra.png',dpi=230);plt.close()
    print(f'Saved per-band diagnostics for {len(ckpts)} checkpoints')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='task',required=True)
    for name in ['audit','bootstrap','transfer-table','spectra']:
        sp=sub.add_parser(name)
        sp.add_argument('--root',type=Path,default=Path('/home/nibio/HSIwheat_40'))
        sp.add_argument('--out',type=Path,default=None)
        if name=='bootstrap':
            sp.add_argument('--nboot',type=int,default=5000)
            sp.add_argument('--seed',type=int,default=2026)
        if name=='spectra':
            sp.add_argument('--max-models',type=int,default=11,help='0 for all')
            sp.add_argument('--device',default='cuda' if __import__('importlib').util.find_spec('torch') else 'cpu')
    a=parser.parse_args()
    root=a.root.expanduser()
    out=(a.out or root/'revision_review2_diagnostics').expanduser();out.mkdir(parents=True,exist_ok=True)
    print('Data root:',root,'\nResults:',out,flush=True)
    if a.task=='audit':audit(root,out)
    elif a.task=='bootstrap':bootstrap(root,out,a.nboot,a.seed)
    elif a.task=='transfer-table':transfer_table(root,out)
    elif a.task=='spectra':spectra(root,out,a.max_models,a.device)

if __name__=='__main__':main()