"""Bounded-memory real-data transfer and training-only-selected within-data study.

Uses compact continuous NPZ + scalar window-index exports. Original Galaxy model
artifacts are never modified. All source-transfer outputs are frozen before
reference joining; independent within-dataset training uses a fixed subject split.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.utils.submission_protocol import (BASELINES, FEATURES, MOTION, QUALITY, fit_expert, fit_gate,
    gate_probability, grouped_splits, metrics, route, select_experts, sha256, window_keys, write_json)
from src.utils.evaluate_external_submission import load_frozen_source, frozen_predictions, validate_unit_contract

SOURCE=Path('experiments/conference_submission_2026_11_01')
SIMPLE=[n for n in BASELINES if n!='pulseppg_random_forest']

def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def log(s): print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {s}',flush=True)
def resolve(manifest_path, stored):
    p=Path(stored)
    return p if p.is_absolute() else (Path(manifest_path).parent/p).resolve()

def participant_windows(index, arrays):
    """Only observed waveform and timestamps reach the encoder/feature API."""
    ppg=arrays['ppg']; fs=float(arrays['ppg_sampling_hz'])
    ts=arrays.get('_ppg_ts')
    if ts is None: ts=arrays['ppg_timestamps_ms'] if 'ppg_timestamps_ms' in arrays else np.arange(len(ppg))*1000/fs
    rows=[]
    for row in index.itertuples(index=False):
        a,b=np.searchsorted(ts,[row.window_start_ms,row.window_end_ms],side='left')
        rows.append(dict(window_key=row.window_key,participant_id=str(row.participant_id),session_id=row.session_id,
            session_name=row.session_name,window_start_ms=row.window_start_ms,window_end_ms=row.window_end_ms,
            window_length_s=10.,stride_s=2.,ppg_sampling_hz=fs,ppg_values=ppg[a:b],ppg_raw_values=ppg[a:b],
            ppg_timestamps_ms=ts[a:b]))
    return pd.DataFrame(rows)

def compute_features(signals, arrays, scale):
    from src.baseline.peak_detection import estimate_hr_from_peaks
    from src.baseline.spectral_hr import estimate_hr_from_spectrum
    from src.utils.build_week3_artifacts import compute_ppg_quality_features,compute_single_acc_window
    acc=arrays['acc']; fs=float(arrays['acc_sampling_hz'])
    ts=arrays.get('_acc_ts')
    if ts is None: ts=arrays['acc_timestamps_ms'] if 'acc_timestamps_ms' in arrays else np.arange(len(acc))*1000/fs
    norm=arrays.get('_acc_norm')
    if norm is None: norm=np.linalg.norm(acc.astype(float)*scale,axis=1)
    rows=[]
    for _,row in signals.iterrows():
        peak=estimate_hr_from_peaks(row.ppg_values,row.ppg_timestamps_ms,sampling_rate_hz=row.ppg_sampling_hz)
        spectral=estimate_hr_from_spectrum(row.ppg_values,row.ppg_timestamps_ms,sampling_rate_hz=row.ppg_sampling_hz)
        a,b=np.searchsorted(ts,[row.window_start_ms,row.window_end_ms],side='left')
        c=peak.predicted_hr_bpm; f=spectral.predicted_hr_bpm
        rows.append({'window_key':row.window_key,'peak_based':c,'spectral':f,
            **compute_ppg_quality_features(row),**compute_single_acc_window(row.window_key,ts[a:b],norm[a:b]),
            'peak_spectral_disagreement_bpm':abs(c-f) if c is not None and f is not None else np.nan})
    return pd.DataFrame(rows)

def prepare_participant(manifest_path, participant, units_path, root, threads=4):
    import torch
    from src.models.common import SignalPreprocessingConfig,build_window_signal_matrix
    from src.models.pulseppg_feature import load_pulseppg_encoder
    from src.models.papagei_feature import load_papagei_encoder
    manifest_path=Path(manifest_path); root=Path(root); out=root/'prepared'/participant
    out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(manifest_path); units=read_json(units_path)
    unit=validate_unit_contract(float(units['acc_scale_to_mps2']),manifest['dataset'],units)
    if not unit['unit_contract_verified'] and units.get('external_input_unit')!='g_assumed_unverified':
        raise ValueError('Verified units or explicit predeclared diagnostic g assumption required')
    stored=manifest['participant_archives'][participant]
    archive=resolve(manifest_path,stored['path'] if isinstance(stored,dict) else stored)
    source=load_frozen_source(SOURCE)
    inputs={'manifest_sha256':sha256(manifest_path),'archive_sha256':sha256(archive),'units_sha256':sha256(units_path),
        'source_freeze_sha256':sha256(SOURCE/'manifests/frozen_fit.json'),'implementation_sha256':sha256(__file__)}
    receipt=out/'complete.json'
    if receipt.exists():
        previous=read_json(receipt)
        if previous['inputs']!=inputs or any(sha256(out/p)!=h for p,h in previous['artifacts'].items()):
            raise ValueError(f'Prepared participant inputs changed: {participant}')
        return previous
    index=pd.read_csv(resolve(manifest_path,manifest['windows_path']),dtype={'participant_id':str})
    index=index.loc[index.participant_id.eq(participant)].reset_index(drop=True)
    index['window_key']=window_keys(index)
    if index.empty or index.window_key.duplicated().any(): raise ValueError('Empty/duplicate participant windows')
    metadata=index[['window_key','participant_id','session_id','session_name','window_start_ms','window_end_ms']]
    references=index[['window_key','label_hr_bpm']].rename(columns={'label_hr_bpm':'y_true'})
    metadata.to_csv(out/'metadata.csv',index=False); references.to_csv(out/'references.csv',index=False)
    arrays=dict(np.load(archive,allow_pickle=False))
    arrays['_ppg_ts']=arrays.get('ppg_timestamps_ms',np.arange(len(arrays['ppg']))*1000/float(arrays['ppg_sampling_hz']))
    arrays['_acc_ts']=arrays.get('acc_timestamps_ms',np.arange(len(arrays['acc']))*1000/float(arrays['acc_sampling_hz']))
    arrays['_acc_norm']=np.linalg.norm(arrays['acc'].astype(float)*float(units['acc_scale_to_mps2']),axis=1)
    torch.set_num_threads(threads)
    feature_blocks=[]; prediction_blocks=[]; n=len(index)
    encoder_models={}; matrices={}
    for name,hz,loader in [('pulseppg',50,load_pulseppg_encoder),('papagei',125,load_papagei_encoder)]:
        encoder_models[name]=loader(checkpoint_path=source.checkpoints[name],device='cpu')[0]
        matrices[name]=np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype='float32',shape=(n,512))
    with torch.inference_mode(), threadpool_limits(limits=threads):
        for start in range(0,n,512):
            signals=participant_windows(index.iloc[start:start+512],arrays)
            features=compute_features(signals,arrays,float(units['acc_scale_to_mps2']))
            emb={}
            for name,hz in [('pulseppg',50),('papagei',125)]:
                pc=source.config['preprocessing']
                config=SignalPreprocessingConfig(mode=pc['mode'],target_sampling_hz=hz,
                    normalization=pc['normalization'],apply_bandpass=pc['apply_bandpass'],low_hz=pc['low_hz'],
                    high_hz=pc['high_hz'],filter_order=pc['filter_order'])
                values=build_window_signal_matrix(signals,hz,preprocessing_config=config)
                outputs=[]
                for offset in range(0,len(values),64):
                    v=encoder_models[name](torch.from_numpy(values[offset:offset+64]).unsqueeze(1))
                    if isinstance(v,(list,tuple)): v=v[0]
                    outputs.append(v.cpu().numpy())
                emb[name]=np.vstack(outputs)
                matrices[name][start:start+len(signals)]=emb[name]
            classical=features[['window_key','peak_based','spectral']]
            predicted=frozen_predictions(signals,classical,features[['window_key']+MOTION+QUALITY],emb,source,manifest['dataset'])
            prediction_blocks.append(predicted)
            feature_blocks.append(features)
            if start%2048==0: log(f"{manifest['dataset']} {participant}: {min(start+512,n)}/{n}")
    for m in matrices.values(): m.flush()
    pd.concat(feature_blocks,ignore_index=True).to_csv(out/'features.csv',index=False)
    pd.concat(prediction_blocks,ignore_index=True).to_csv(out/'frozen_predictions_without_labels.csv',index=False)
    result={'participant':participant,'n_windows':n,'inputs':inputs,'encoder_frozen':True,'no_external_fit':True,
        'artifacts':{p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}}
    write_json(receipt,result); log(f'Prepared {participant}: {n} windows')
    return result

def summarize_predictions(wide, names, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    common=np.isfinite(wide[names+['y_true']].to_numpy(float)).all(axis=1)
    write_json(out/'coverage.json',{'eligible_windows':len(wide),'paired_windows':int(common.sum()),
        'eligible_participants':wide.participant_id.nunique(),'paired_participants':wide.loc[common].participant_id.nunique(),
        'failures_by_model':{name:int((~np.isfinite(wide[name])).sum()) for name in names},'mask':'common_finite_only'})
    frame=wide.loc[common].copy()
    if frame.empty: raise ValueError('No common finite predictions')
    frame.to_csv(out/'paired_predictions.csv',index=False)
    summaries=[]; participants=[]
    for name in names:
        for pid,group in frame.groupby('participant_id'):
            participants.append({'model_name':name,'participant_id':pid,**metrics(group.y_true,group[name])})
        per=pd.DataFrame(participants).query('model_name == @name')
        summaries.append({'model_name':name,**metrics(frame.y_true,frame[name]),'n_participants':frame.participant_id.nunique(),
            'participant_mean_MAE':float(per.MAE.mean()),'participant_std_MAE':float(per.MAE.std())})
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(participants).to_csv(out/'participant_metrics.csv',index=False)
    for column in ['activity','motion_intensity_bin','ppg_quality_bin']:
        if column not in frame: continue
        rows=[{'model_name':name,column:value,**metrics(group.y_true,group[name]),'n_participants':group.participant_id.nunique()}
              for value,group in frame.groupby(column,dropna=False) for name in names]
        pd.DataFrame(rows).to_csv(out/f'by_{column}.csv',index=False)
    return frame,pd.DataFrame(summaries)

def transfer(manifest_path,root,units_path):
    manifest=read_json(manifest_path); root=Path(root)
    parts=[]; receipt_hashes={}
    for pid in manifest['participant_archives']:
        out=root/'prepared'/pid; receipt=read_json(out/'complete.json')
        for p,h in receipt['artifacts'].items():
            if sha256(out/p)!=h: raise ValueError('Prepared frozen inference artifact modified')
        prediction=pd.read_csv(out/'frozen_predictions_without_labels.csv')
        reference=pd.read_csv(out/'references.csv')
        reference.window_key=manifest['dataset']+'::'+reference.window_key
        parts.append(prediction.merge(reference,on='window_key',validate='one_to_one'))
        receipt_hashes[pid]=sha256(out/'complete.json')
    wide=pd.concat(parts,ignore_index=True)
    names=BASELINES+[f'{feature}_{kind}' for feature in FEATURES for kind in ['hard','soft']]+['best_classical','best_foundation','best_single']
    frame,summary=summarize_predictions(wide,names,root/'transfer')
    from src.utils.submission_statistics import bootstrap_participant_mean,signed_rank_pvalue
    per=pd.read_csv(root/'transfer/participant_metrics.csv'); stats=[]
    for name in [f'{f}_{k}' for f in FEATURES for k in ['hard','soft']]:
        b=per.loc[per.model_name.eq('best_single')].set_index('participant_id')
        r=per.loc[per.model_name.eq(name)].set_index('participant_id').loc[b.index]
        for m in ['MAE','RMSE','p95_absolute_error','catastrophic_error_rate_20bpm']:
            delta=b[m]-r[m]; lo,hi=bootstrap_participant_mean(delta)
            stats.append({'router':name,'metric':m,'n_participants':len(delta),'mean_improvement':delta.mean(),
                'ci95_low':lo,'ci95_high':hi,'signed_rank_p':signed_rank_pvalue(delta),
                'improved':int((delta>0).sum()),'degraded':int((delta<0).sum())})
    pd.DataFrame(stats).to_csv(root/'transfer/participant_statistics.csv',index=False)
    write_json(root/'transfer/transfer_manifest.json',{'status':'real_frozen_transfer_complete','dataset':manifest['dataset'],
        'source_frozen_fit_sha256':sha256(SOURCE/'manifests/frozen_fit.json'),'dataset_manifest_sha256':sha256(manifest_path),
        'units':read_json(units_path),'prepared_receipts':receipt_hashes,'no_external_fit_or_selection':True,
        'motion_routing_unit_contract_verified':bool(read_json(units_path)['external_units_verified']),
        'unit_independent_methods':BASELINES+['quality_hard','quality_soft'],
        'interpretation':'Motion-involving transfer is conditional on predeclared unit assumption if unverified; do not claim fully verified motion transfer.',
        'pretraining_overlap_audit':'reports/external_data_sources_2026_11_01.md','all_external_participants_evaluated':True,
        'artifacts':{p.name:sha256(p) for p in (root/'transfer').iterdir() if p.is_file() and p.name!='transfer_manifest.json'}})
    log(summary[['model_name','MAE','RMSE','n_windows']].to_string(index=False))

def within(manifest_path,root):
    """The fixed subject holdout never enters expert tuning or gate supervision."""
    import math,random
    root=Path(root); manifest=read_json(manifest_path); out=root/'within_dataset'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'within_manifest.json').exists():
        previous=read_json(out/'within_manifest.json')
        if any(sha256(out/p)!=h for p,h in previous['artifacts'].items()): raise ValueError('Within results changed')
        return
    people=sorted(manifest['participant_archives']); shuffled=people.copy(); random.Random(42).shuffle(shuffled)
    testing=sorted(shuffled[:max(2,math.ceil(.2*len(people)))]); training=sorted(set(people)-set(testing))
    split={'seed':42,'train_participants':training,'test_participants':testing,'selection_source':'training_grouped_oof_only'}
    split_path=out/'participant_split.json'
    if split_path.exists() and read_json(split_path)!=split: raise ValueError('Within split changed')
    write_json(split_path,split)
    frames=[]; embeddings={m:[] for m in ['pulseppg','papagei']}
    for pid in people:
        part=root/'prepared'/pid
        meta=pd.read_csv(part/'metadata.csv',dtype={'participant_id':str})
        features=pd.read_csv(part/'features.csv')
        refs=pd.read_csv(part/'references.csv')
        frames.append(meta.merge(features,on='window_key',validate='one_to_one').merge(refs,on='window_key',validate='one_to_one'))
        for name in embeddings: embeddings[name].append(np.load(part/f'{name}.npy'))
    data=pd.concat(frames,ignore_index=True)
    x={name:np.concatenate(values) for name,values in embeddings.items()}; del embeddings
    train_mask=data.participant_id.isin(training).to_numpy(); test_mask=~train_mask
    tr=data.loc[train_mask].reset_index(drop=True); te=data.loc[test_mask].drop(columns='y_true').reset_index(drop=True)
    y=tr.y_true.to_numpy(float); groups=tr.participant_id.to_numpy()
    oof=tr.copy(); oof['fold']=-1
    for name in SIMPLE[2:]: oof[name]=np.nan
    records=[]
    with threadpool_limits(limits=4):
        for fold,(a,b) in enumerate(grouped_splits(groups,5),1):
            held=sorted(set(groups[b])); record={'fold':fold,'train_participants':sorted(set(groups[a])),
                'validation_participants':held,'experts':{}}
            oof.loc[b,'fold']=fold
            for name in SIMPLE[2:]:
                model,kind=name.split('_',1); values=np.asarray(x[model][train_mask],dtype=float)
                log(f"{manifest['dataset']} within fold {fold}/5 {name}")
                fitted,fit_record=fit_expert(values[a],y[a],groups[a],kind,42,training,held,3)
                oof.loc[b,name]=fitted.predict(values[b]); record['experts'][name]=fit_record
            records.append(record)
        if oof.groupby('participant_id').fold.nunique().ne(1).any() or not set(oof.participant_id)<=set(training):
            raise ValueError('Within OOF provenance leakage')
        for record in records:
            if set(record['train_participants']) & set(record['validation_participants']): raise ValueError('OOF overlap')
            for fitted in record['experts'].values():
                if set(fitted['fit_participants'])!=set(record['train_participants']): raise ValueError('Expert provenance mismatch')
        selection=select_experts(oof,training,candidates=SIMPLE)
        write_json(out/'selection.json',selection); write_json(out/'oof_provenance.json',records)
        oof.to_csv(out/'training_oof_predictions.csv',index=False)
        final_records={}
        for name in SIMPLE[2:]:
            model,kind=name.split('_',1)
            log(f"{manifest['dataset']} within final fit {name}")
            fitted,record=fit_expert(np.asarray(x[model][train_mask],float),y,groups,kind,42,training,testing,5)
            te[name]=fitted.predict(np.asarray(x[model][test_mask],float))
            joblib.dump(fitted,out/f'{name}.joblib'); final_records[name]=record
        c,f=selection['best_classical'],selection['best_foundation']
        valid=np.isfinite(oof[c])&np.isfinite(oof[f]); gt=oof.loc[valid]
        target=((gt[f]-gt.y_true).abs()<(gt[c]-gt.y_true).abs()).astype(int).to_numpy()
        gt.to_csv(out/'router_training_data.csv',index=False)
        for feature,columns in FEATURES.items():
            gate=fit_gate(gt,target,columns,training,42); joblib.dump(gate,out/f'{feature}_gate.joblib')
            p=gate_probability(gate,te,columns)
            for soft in [False,True]: te[feature+('_soft' if soft else '_hard')]=route(te[c],te[f],p,soft)
        for alias in ['best_classical','best_foundation','best_single']: te[alias]=te[selection[alias]]
    write_json(out/'final_fit_provenance.json',final_records)
    te['activity']=te.session_name
    te.to_csv(out/'frozen_predictions_without_labels.csv',index=False)
    frozen={'freeze_at_utc':datetime.now(timezone.utc).isoformat(),'test_participants':testing,
        'training_participants':training,'test_labels_used_for_fit':False,
        'frozen_artifacts':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/'frozen_fit.json',frozen)
    # Only evaluation joins the held-out references after predictor serialization.
    references=data.loc[test_mask,['window_key','y_true']]
    final=te.merge(references,on='window_key',validate='one_to_one')
    final['oracle']=np.where((final[c]-final.y_true).abs()<=(final[f]-final.y_true).abs(),final[c],final[f])
    names=SIMPLE+[f'{s}_{k}' for s in FEATURES for k in ['hard','soft']]+['best_classical','best_foundation','best_single','oracle']
    _,summary=summarize_predictions(final,names,out)
    write_json(out/'within_manifest.json',{'status':'real_within_dataset_subject_holdout_complete','dataset':manifest['dataset'],
        'split':split,'selection':selection,'preprocessing':'unchanged_harmonized_encoder_inputs',
        'strong_regressor_note':'RF source-transfer evaluated separately; within-data core probes are linear/ridge',
        'artifacts':{p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='within_manifest.json'}})
    log(summary[['model_name','MAE','RMSE','n_windows']].to_string(index=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True); parser.add_argument('--units',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','transfer','within','all'],default='all')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    manifest=read_json(args.manifest)
    if manifest.get('window_storage')!='continuous_npz_index_v1': raise ValueError('Compact data contract required')
    config={'manifest':str(args.manifest),'manifest_sha256':sha256(args.manifest),'units_sha256':sha256(args.units),
        'source_freeze_sha256':sha256(SOURCE/'manifests/frozen_fit.json'),'workers':args.workers,'seed':42,
        'within_protocol':'participant_holdout_seed42; outer5_inner3; linear_ridge_only; gate_C1_threshold0.5',
        'preprocessing':'unchanged_Galaxy_harmonized', 'primary_router':'motion_quality_soft'}
    cp=args.output_root/'frozen_config.json'
    if cp.exists() and read_json(cp)!=config: raise ValueError('Experiment protocol changed')
    write_json(cp,config)
    if args.stage in ['prepare','all']:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(prepare_participant,str(args.manifest),p,str(args.units),str(args.output_root)) for p in manifest['participant_archives']]
            for future in as_completed(futures): log(f"Completed worker {future.result()['participant']}")
    if args.stage in ['transfer','all']: transfer(args.manifest,args.output_root,args.units)
    if args.stage in ['within','all']: within(args.manifest,args.output_root)

if __name__=='__main__': main()
