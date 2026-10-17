"""Stage 5 checkpointed repeated study. python full_study.py run|check|summarize"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
import run_experiment as b
import continue_project as s

ROOT=b.ROOT/'results'/'full_study'
METHODS=['tfidf_lr','minilm_proto']


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def environment():
    return {'python':platform.python_version(),'versions':{
        n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy']}}


def provenance(method,seed,phase,adapt=None,shots=None):
    model_functions=[b.fit_tfidf_lr,b.lr_predict] if method=='tfidf_lr' else [s.make_prototypes,s.predict_proto]
    return {'method':method,'split_seed':seed,'phase':phase,'adapt_seed':adapt,'shots':shots,
        'split_sha256':b.sha256(b.split_path(seed)), 'dataset_fingerprint':b.fingerprint(),
        'model_config':b.CFG['few_shot_tfidf_lr'] if method=='tfidf_lr' else s.cache_spec(),
        'cache_meta_sha256':None if method=='tfidf_lr' else b.sha256(s.CACHE/'meta.json'),
        'threshold_config':b.CFG['threshold'],
        'fit_prediction_hash':hashlib.sha256(''.join(inspect.getsource(f) for f in model_functions).encode()).hexdigest(),
        'phase_runner_hash':hashlib.sha256(inspect.getsource(compute).encode()).hexdigest(),
        'threshold_hash':hashlib.sha256(inspect.getsource(b.pick_threshold).encode()).hexdigest()}


def folder(method,seed,phase,adapt=None,shots=None):
    tag='pre' if phase=='pre' else f'post_a{adapt}_k{shots}'
    return ROOT/'runs'/method/f'split{seed}'/tag


def save_phase(path,meta,pred,dev,info):
    path.mkdir(parents=True,exist_ok=True)
    pred.to_csv(path/'predictions.csv',index=False)
    dev.to_csv(path/'dev_scores.csv',index=False)
    s.write_json(path/'manifest.json',{'meta':meta,'info':info,
        'predictions_sha256':b.sha256(path/'predictions.csv'),
        'dev_scores_sha256':b.sha256(path/'dev_scores.csv')})


def verify_phase(path,meta):
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['meta']==meta,'stale phase metadata'
    for file,key in [('predictions.csv','predictions_sha256'),('dev_scores.csv','dev_scores_sha256')]:
        assert b.sha256(path/file)==manifest[key],file+' checksum'
    pred=pd.read_csv(path/'predictions.csv');dev=pd.read_csv(path/'dev_scores.csv')
    train,test=b.load_prepared();d=b.load_split(meta['split_seed']);C=d['classes'];phase=meta['phase']
    assert set(pred.stage)=={phase} and set(dev.stage)=={phase}
    assert pred.accepted.dtype==bool and np.isfinite(pred.score).all() and np.isfinite(dev.score).all()
    assert set(pred.population)=={'orig_supported','new','test_unknown'}
    expected_parts=['dev_supported']+([] if phase=='pre' else ['dev_new'])+['dev_unknown']
    assert set(dev.population)==set(expected_parts)
    for part in expected_parts:
        rows=dev[dev.population==part].row
        assert len(rows)==len(d[part+'_rows']) and set(rows)==set(d[part+'_rows'])
    cal=dev[dev.population!='dev_unknown'];thr=b.pick_threshold(cal.score)
    assert np.allclose(pred.threshold,thr,atol=1e-12,rtol=0)
    assert ((pred.score>=thr)==pred.accepted).all()
    info=manifest['info']
    assert np.isclose(info['threshold'],thr,atol=1e-12,rtol=0)
    assert np.isclose(info['dev_supported_acceptance'],(cal.score>=thr).mean(),atol=1e-12,rtol=0)
    unk=dev[dev.population=='dev_unknown']
    assert np.isclose(info['dev_unknown_false_acceptance'],(unk.score>=thr).mean(),atol=1e-12,rtol=0)
    allowed=C['supported']+([] if phase=='pre' else C['new'])
    assert set(pred.pred)<=set(allowed)
    for pop,cls in [('orig_supported','supported'),('new','new'),('test_unknown','test_unknown')]:
        x=pred[pred.population==pop];expected=test[test.category.isin(C[cls])]
        assert len(x)==len(expected) and set(x.row)==set(expected.row)
        assert (x.true.to_numpy()==test.iloc[x.row].category.to_numpy()).all()
        assert (x.text.to_numpy()==test.iloc[x.row].clean.to_numpy()).all()
    return pred,dev,manifest


def compute(method,seed,phase,E,adapt=None,shots=None):
    train,test=b.load_prepared();d=b.load_split(seed);C=d['classes']
    original=d['train_supported_rows'];new=[]
    if phase=='post':new=[r for cls in C['new'] for r in d['adapt'][str(adapt)][cls][:shots]]
    assert phase=='pre' or len(new)==len(C['new'])*shots
    caught=[];start=time.perf_counter()
    if method=='tfidf_lr':
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model=b.fit_tfidf_lr(train.iloc[original+new].clean,train.iloc[original+new].category)
        def predict(part,rows):
            frame=train if part=='train' else test
            return b.lr_predict(model,frame.iloc[rows].clean)
        iterations=int(model[1].n_iter_.max())
    else:
        model=s.make_prototypes(E['train'],original,train.category)
        if new:
            extra=s.make_prototypes(E['train'],new,train.category)
            labels=np.concatenate([model[0],extra[0]]);vectors=np.vstack([model[1],extra[1]])
            order=np.argsort(labels);model=(labels[order],vectors[order])
        def predict(part,rows):return s.predict_proto(model,E[part][rows])
        iterations=None
    elapsed=time.perf_counter()-start
    dev=[]
    for pop in ['dev_supported']+([] if phase=='pre' else ['dev_new'])+['dev_unknown']:
        rows=d[pop+'_rows'];_,scores=predict('train',rows)
        dev.extend({'stage':phase,'population':pop,'row':r,'score':float(score)} for r,score in zip(rows,scores))
    dev=pd.DataFrame(dev);cal=dev[dev.population!='dev_unknown'];thr=b.pick_threshold(cal.score)
    pred=[]
    for pop,group in [('orig_supported','supported'),('new','new'),('test_unknown','test_unknown')]:
        frame=test[test.category.isin(C[group])];rows=frame.row.to_numpy();labels,scores=predict('test',rows)
        pred.append(pd.DataFrame({'stage':phase,'population':pop,'row':rows,'text':frame.clean.to_numpy(),
            'true':frame.category.to_numpy(),'pred':labels,'score':scores,'accepted':scores>=thr,'threshold':thr}))
    info={'origin':'executed_stage5','environment':environment(),'fit_seconds':elapsed,
        'fit_runtime_scope':'LR refits TF-IDF and classifier; prototypes form original and new prototypes using cached vectors',
        'new_fit_examples':len(new),'additional_new_dev_examples':0 if phase=='pre' else len(d['dev_new_rows']),
        'threshold':thr,'calibration_count':len(cal),'dev_supported_acceptance':float((cal.score>=thr).mean()),
        'dev_unknown_false_acceptance':float((dev[dev.population=='dev_unknown'].score>=thr).mean()),
        'max_iterations_used':iterations,'warnings':[str(w.message) for w in caught]}
    return pd.concat(pred,ignore_index=True),dev,info


def import_pilot(method,seed,phase,adapt,shots,meta,path):
    if seed!=0 or (phase=='post' and (adapt!=0 or shots!=5)):return False
    src=b.SMALL_DIR if method=='tfidf_lr' else s.PROTO
    report='metrics_corrected.json' if method=='tfidf_lr' else 'metrics.json'
    res=json.loads((src/report).read_text())
    pred=pd.read_csv(src/'predictions.csv');pred=pred[pred.stage==phase]
    dev=pd.read_csv(src/'dev_scores.csv');dev=dev[dev.stage==phase]
    cal=dev[dev.population!='dev_unknown'];thr=b.pick_threshold(cal.score)
    prov=res['threshold_provenance'][phase]
    assert np.isclose(thr,prov['threshold'],atol=1e-12,rtol=0)
    info={'origin':'reused_verified_pilot','pilot_report_sha256':b.sha256(src/report),
        'pilot_prediction_sha256':b.sha256(src/'predictions.csv'),'pilot_source':str(src.relative_to(b.ROOT)),
        'environment_note':'Pilot environment recorded in Stage 3/4 state; LR pilot sklearn 1.9.1, new runs record actual environment.',
        'fit_seconds':res['initial_fit_seconds'] if phase=='pre' else
            res.get('adaptation_seconds_refit_tfidf_and_lr',res.get('adaptation_seconds_cached_prototype_update')),
        'fit_runtime_scope':'Historical pilot runtime; not directly comparable across environments or cached/original prototype construction',
        'new_fit_examples':0 if phase=='pre' else 75,
        'additional_new_dev_examples':0 if phase=='pre' else len(b.load_split(seed)['dev_new_rows']),
        'threshold':thr,'calibration_count':len(cal),
        'dev_supported_acceptance':float((cal.score>=thr).mean()),
        'dev_unknown_false_acceptance':float((dev[dev.population=='dev_unknown'].score>=thr).mean()),
        'warnings':[],'warning_note':'Historical pilot did not save solver warnings.'}
    save_phase(path,meta,pred,dev,info);return True


def enumerate_runs():
    for method in METHODS:
        for seed in b.CFG['splits']['split_seeds']:
            yield method,seed,'pre',None,None
            for adapt in b.CFG['splits']['adapt_seeds']:
                for shots in b.CFG['splits']['shots']:yield method,seed,'post',adapt,shots


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    b.check_splits();s.check();E,_=s.load_embeddings()
    s.write_json(ROOT/'environment.json',environment())
    for i,params in enumerate(enumerate_runs(),1):
        method,seed,phase,adapt,shots=params;path=folder(*params);meta=provenance(*params)
        if (path/'manifest.json').exists():
            verify_phase(path,meta);action='reused existing checkpoint'
        elif import_pilot(method,seed,phase,adapt,shots,meta,path):action='imported verified pilot'
        else:
            with threadpool_limits(limits=1):pred,dev,info=compute(method,seed,phase,E,adapt,shots)
            save_phase(path,meta,pred,dev,info);action='executed'
        verify_phase(path,meta)
        print(f'{i}/60 {method} split={seed} {phase} adapt={adapt} shots={shots}: {action}',flush=True)
    check();summarize()


def check():
    b.check_splits();count=0
    for params in enumerate_runs():verify_phase(folder(*params),provenance(*params));count+=1
    assert count==60
    print('Verified 60 phases: six independent pre baselines + 54 post configurations.',flush=True)


def summarize():
    rows=[]
    for method in METHODS:
        for seed in b.CFG['splits']['split_seeds']:
            pre,_,_=verify_phase(folder(method,seed,'pre'),provenance(method,seed,'pre'))
            for adapt in b.CFG['splits']['adapt_seeds']:
                for shots in b.CFG['splits']['shots']:
                    post,_,manifest=verify_phase(folder(method,seed,'post',adapt,shots),provenance(method,seed,'post',adapt,shots))
                    metrics=s.metrics(pd.concat([pre,post]),b.load_split(seed)['classes'])
                    row={'method':method,'split_seed':seed,'adapt_seed':adapt,'shots':shots,
                        'metric_code_hash':s.metric_identity(),'origin':manifest['info']['origin']}
                    for stage,values in metrics.items():
                        for key,value in values.items():row[f'{stage}__{key}']=value
                    rows.append(row)
    df=pd.DataFrame(rows);assert len(df)==54
    df.to_csv(ROOT/'metrics_per_run.csv',index=False)
    numeric=[c for c in df if c.startswith(('pre__','post__','change_after_addition__'))]
    # Average sampling seeds within each split; then describe variation across 3 splits.
    split=df.groupby(['method','shots','split_seed'])[numeric].mean().reset_index()
    split.to_csv(ROOT/'metrics_per_split.csv',index=False)
    mean=split.groupby(['method','shots'])[numeric].mean()
    std=split.groupby(['method','shots'])[numeric].std(ddof=1)
    summary=pd.concat([mean.add_suffix('__mean'),std.add_suffix('__split_sd')],axis=1).reset_index()
    summary.to_csv(ROOT/'summary.csv',index=False)
    s.write_json(ROOT/'summary_meta.json',{'metric_code_hash':s.metric_identity(),
        'aggregation':'Mean over 3 adaptation samples within split; equal-weight average over 3 category splits. SD across split means is descriptive, not an independent-dataset confidence interval.',
        'completed_post_configurations':54,'completed_pre_baselines':6,
        'calibration_budget':'k-shot fitting plus additional labelled new-intent dev calibration',
        'mixed_environment_note':'Verified pilot predictions reused; LR pilot sklearn 1.9.1 versus Stage 5 environment recorded separately. Do not claim exact single-environment replication.',
        'no_test_tuning':True})
    columns=['method','shots','post__new_message_subset_accuracy__mean',
        'post__all_supported_closed_set_macro_f1__mean','post__test_unknown_false_acceptance__mean']
    print(summary[columns].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','check','summarize'])
    args=parser.parse_args();{'run':run,'check':check,'summarize':summarize}[args.command]()
