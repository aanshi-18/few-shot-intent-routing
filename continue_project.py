"""Stage 4: refresh metrics without refitting LR, cache frozen embeddings, run prototypes.
Run from this directory: python continue_project.py refresh-metrics|embed|run-small-proto|check
"""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import run_experiment as base

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data' / 'embeddings'
PROTO = ROOT / 'results' / 'small_run_proto'
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
# Public repository commit observed at the beginning of this stage.
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def metric_identity():
    import inspect
    text = inspect.getsource(base.metrics_from_predictions) + inspect.getsource(base.macro_f1_manual)
    text += inspect.getsource(metrics)
    return hashlib.sha256(text.encode()).hexdigest()


def metrics(predictions, classes):
    out = base.metrics_from_predictions(predictions, classes)
    for stage in ['pre', 'post']:
        m = out[stage]
        m['original_message_subset_macro_f1'] = m.pop('closed_set_macro_f1_original_intents')
        m['original_message_subset_accuracy'] = m.pop('closed_set_accuracy_original_intents')
        p = predictions[(predictions.stage == stage) & predictions.population.isin(
            ['orig_supported'] if stage == 'pre' else ['orig_supported', 'new'])]
        for group in (['supported'] if stage == 'pre' else ['supported', 'new']):
            key = 'original' if group == 'supported' else 'new'
            m[f'{key}_category_macro_f1_on_all_supported_messages'] = float(f1_score(
                p.true, p.pred, labels=classes[group], average='macro', zero_division=0))
        labels = classes['supported'] + ([] if stage == 'pre' else classes['new'])
        m['all_supported_closed_set_macro_f1'] = float(f1_score(
            p.true, p.pred, labels=labels, average='macro', zero_division=0))
        if stage == 'post':
            m['new_message_subset_macro_f1'] = m.pop('closed_set_macro_f1_new_intents')
            m['new_message_subset_accuracy'] = m.pop('closed_set_accuracy_new_intents')
    out['change_after_addition']['original_message_subset_macro_f1'] = out['change_after_addition'].pop('original_macro_f1')
    out['change_after_addition']['original_category_macro_f1_on_all_supported_messages'] = (
        out['post']['original_category_macro_f1_on_all_supported_messages'] -
        out['pre']['original_category_macro_f1_on_all_supported_messages'])
    return out


def refresh_metrics():
    """Keep historical metrics intact; derive corrected report from existing predictions."""
    base.check_splits()
    base.check_small_run()
    old = json.loads((base.SMALL_DIR / 'metrics.json').read_text())
    pred = pd.read_csv(base.SMALL_DIR / 'predictions.csv')
    classes = base.load_split(old['meta']['split_seed'])['classes']
    res = dict(old)
    res['metrics'] = metrics(pred, classes)
    res['reporting_meta'] = {
        'metric_code_hash': metric_identity(),
        'predictions_sha256': base.sha256(base.SMALL_DIR / 'predictions.csv'),
        'original_metrics_sha256': base.sha256(base.SMALL_DIR / 'metrics.json'),
        'calibration_budget': 'Additional labelled dev messages for new classes; not strict k-shot end-to-end.'}
    write_json(base.SMALL_DIR / 'metrics_corrected.json', res)
    print(json.dumps(res['metrics'], indent=2))


def cache_spec():
    import inspect
    return {'encoder': ENCODER, 'revision': REVISION,
            'dataset_fingerprint': base.fingerprint(), 'normalization': 'L2',
            'preprocessing': 'whitespace collapsed; case retained',
            'clean_text_sha256': hashlib.sha256(inspect.getsource(base.clean_text).encode()).hexdigest(),
            'max_seq_length': 256, 'precision': 'float32'}


def load_embeddings():
    info = json.loads((CACHE / 'meta.json').read_text())
    if info['spec'] != cache_spec():
        raise RuntimeError('Stale embedding cache: data, encoder or preprocessing changed.')
    arrays = {}
    for part, n in [('train', 10003), ('test', 3080)]:
        path = CACHE / f'{part}.npy'
        if base.sha256(path) != info['files'][part]:
            raise RuntimeError(f'{part} embedding file checksum mismatch')
        x = np.load(path, allow_pickle=False)
        assert x.shape == (n, 384) and np.isfinite(x).all()
        assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
        arrays[part] = x
    return arrays, info


def embed():
    if (CACHE / 'meta.json').exists():
        _, _ = load_embeddings()
        print('Verified existing embeddings; reused without inference.'); return
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    train, test = base.load_prepared()
    model = SentenceTransformer(ENCODER, revision=REVISION, device='cpu')
    model.max_seq_length = 256
    model.eval()
    CACHE.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    counts = {}
    # Independent frozen inference on each text. No corpus fitting or test-label use.
    for part, frame in [('train', train), ('test', test)]:
        lengths = [len(model.tokenizer.encode(t, add_special_tokens=True)) for t in frame.clean]
        counts[part] = sum(n > model.max_seq_length for n in lengths)
        x = model.encode(frame.clean.tolist(), batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
        np.save(CACHE / f'{part}.npy', x.astype(np.float32), allow_pickle=False)
    info = {'spec': cache_spec(), 'encoding_seconds': time.perf_counter() - start,
            'truncated_text_counts': counts,
            'files': {p: base.sha256(CACHE / f'{p}.npy') for p in ['train', 'test']},
            'versions': {n: importlib.metadata.version(n) for n in
                         ['sentence-transformers', 'torch', 'transformers', 'numpy']}}
    write_json(CACHE / 'meta.json', info)
    load_embeddings()
    print('Real frozen encoder cache built and verified:', info)


def make_prototypes(x, rows, labels):
    ids = np.asarray(rows, dtype=int)
    classes = np.array(sorted(set(labels.iloc[ids])))
    vectors = np.stack([x[ids[labels.iloc[ids].to_numpy() == c]].mean(axis=0) for c in classes])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Near-zero class prototype')
    return classes, vectors / norms


def predict_proto(model, x):
    classes, vectors = model
    similarity = x @ vectors.T
    winners = similarity.argmax(axis=1)
    return classes[winners], similarity[np.arange(len(x)), winners]


def proto_identity():
    import inspect
    return hashlib.sha256(''.join(inspect.getsource(f) for f in
        [make_prototypes, predict_proto, small_proto, base.pick_threshold]).encode()).hexdigest()


def small_proto():
    base.check_splits()
    embeddings, cache = load_embeddings()
    train, test = base.load_prepared()
    c = base.CFG['small_run']; d = base.load_split(c['split_seed']); classes = d['classes']
    meta = {'run': c, 'split_sha256': base.sha256(base.split_path(c['split_seed'])),
            'cache_meta_sha256': base.sha256(CACHE / 'meta.json'),
            'prototype_code_hash': proto_identity(), 'metric_code_hash': metric_identity(),
            'threshold': base.CFG['threshold']}
    PROTO.mkdir(parents=True, exist_ok=True)
    if (PROTO / 'metrics.json').exists():
        existing = json.loads((PROTO / 'metrics.json').read_text())
        if existing['meta'] == meta:
            check(); print('Verified prototype small run reused.'); return
    labels = train.category
    original_rows = d['train_supported_rows']
    adaptation = [r for cls in classes['new'] for r in d['adapt'][str(c['adapt_seed'])][cls][:c['shots']]]
    begin = time.perf_counter()
    original = make_prototypes(embeddings['train'], original_rows, labels)
    original_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    new = make_prototypes(embeddings['train'], adaptation, labels)
    expanded = (np.concatenate([original[0], new[0]]), np.vstack([original[1], new[1]]))
    order = np.argsort(expanded[0]); expanded = (expanded[0][order], expanded[1][order])
    adaptation_seconds = time.perf_counter() - begin
    for cls, vector in zip(original[0], original[1]):
        assert np.array_equal(vector, expanded[1][np.where(expanded[0] == cls)[0][0]])
    predictions, dev_log, provenance = [], [], {}
    for stage, model in [('pre', original), ('post', expanded)]:
        dev_parts = ['dev_supported'] + ([] if stage == 'pre' else ['dev_new'])
        calibration = []
        for part in dev_parts + ['dev_unknown']:
            rows = d[f'{part}_rows']
            _, scores = predict_proto(model, embeddings['train'][rows])
            dev_log.extend({'stage': stage, 'population': part, 'row': r, 'score': float(v)}
                           for r, v in zip(rows, scores))
            if part in dev_parts: calibration.extend(scores)
        threshold = base.pick_threshold(calibration)
        provenance[stage] = {'threshold': threshold, 'n_calibration': len(calibration),
                             'calibration_population': dev_parts,
                             'achieved_dev_supported_acceptance': float((np.array(calibration) >= threshold).mean()),
                             'uses_official_test': False}
        for part, group in [('orig_supported','supported'), ('new','new'), ('test_unknown','test_unknown')]:
            rows = test.loc[test.category.isin(classes[group]), 'row'].to_numpy()
            predicted, scores = predict_proto(model, embeddings['test'][rows])
            predictions.append(pd.DataFrame({'stage': stage, 'population': part, 'row': rows,
                'text': test.iloc[rows].clean.to_numpy(), 'true': test.iloc[rows].category.to_numpy(),
                'pred': predicted, 'score': scores, 'accepted': scores >= threshold, 'threshold': threshold}))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(PROTO / 'predictions.csv', index=False)
    pd.DataFrame(dev_log).to_csv(PROTO / 'dev_scores.csv', index=False)
    for stage in ['pre','post']:
        scores = np.array([v['score'] for v in dev_log if v['stage']==stage and v['population']=='dev_unknown'])
        provenance[stage]['dev_unknown_false_acceptance'] = float((scores >= provenance[stage]['threshold']).mean())
    res = {'meta': meta, 'threshold_provenance': provenance, 'n_adaptation_examples': len(adaptation),
           'initial_fit_seconds': original_seconds, 'adaptation_seconds_cached_prototype_update': adaptation_seconds,
           'encoding_seconds_all_texts_once': cache['encoding_seconds'],
           'runtime_note': 'Prototype update excludes encoding, since embeddings are cached. LR refit excludes no encoder. Report both costs.',
           'calibration_budget': 'Additional labelled new-intent development data.',
           'metrics': metrics(predictions, classes)}
    write_json(PROTO / 'metrics.json', res)
    check()
    print(json.dumps(res['metrics'], indent=2))


def check():
    """Independent saved-output integrity checks for both model families."""
    base.check_splits()
    train, test = base.load_prepared()
    d = base.load_split(base.CFG['small_run']['split_seed']); classes = d['classes']
    for folder, metric_file in [(base.SMALL_DIR, 'metrics_corrected.json'), (PROTO, 'metrics.json')]:
        if not (folder / metric_file).exists(): continue
        res = json.loads((folder / metric_file).read_text())
        pred = pd.read_csv(folder / 'predictions.csv'); dev = pd.read_csv(folder / 'dev_scores.csv')
        assert set(pred.stage) == {'pre','post'} and pred.accepted.dtype == bool
        assert np.isfinite(pred.score).all() and np.isfinite(dev.score).all()
        for stage in ['pre','post']:
            for part, group in [('orig_supported','supported'),('new','new'),('test_unknown','test_unknown')]:
                subset = pred[(pred.stage==stage)&(pred.population==part)]
                expected = test.loc[test.category.isin(classes[group])]
                assert len(subset)==len(expected) and set(subset.row)==set(expected.row)
                assert (subset.true.to_numpy()==test.iloc[subset.row].category.to_numpy()).all()
                assert (subset.text.to_numpy()==test.iloc[subset.row].clean.to_numpy()).all()
                allowed = classes['supported'] + ([] if stage=='pre' else classes['new'])
                assert set(subset.pred) <= set(allowed)
            cal_parts = ['dev_supported']+([] if stage=='pre' else ['dev_new'])
            for part in cal_parts+['dev_unknown']:
                rows=dev[(dev.stage==stage)&(dev.population==part)].row
                assert len(rows)==len(d[f'{part}_rows']) and set(rows)==set(d[f'{part}_rows'])
            cal=dev[(dev.stage==stage)&dev.population.isin(cal_parts)]
            threshold=base.pick_threshold(cal.score)
            assert np.isclose(threshold,res['threshold_provenance'][stage]['threshold'],atol=1e-12,rtol=0)
            subset=pred[pred.stage==stage]
            assert np.allclose(subset.threshold,threshold,atol=1e-12,rtol=0)
            assert ((subset.score>=threshold)==subset.accepted).all()
        again=metrics(pred,classes)
        for stage in again:
            assert set(again[stage])==set(res['metrics'][stage])
            for key,value in again[stage].items():
                saved=res['metrics'][stage][key]
                assert (value is None and saved is None) or np.isclose(value,saved,atol=1e-12,rtol=0),key
        if folder==base.SMALL_DIR:
            assert res['reporting_meta']['metric_code_hash']==metric_identity()
            assert res['reporting_meta']['predictions_sha256']==base.sha256(folder/'predictions.csv')
        else:
            assert res['meta']['metric_code_hash']==metric_identity()
            assert res['meta']['prototype_code_hash']==proto_identity()
            assert res['meta']['cache_meta_sha256']==base.sha256(CACHE/'meta.json')
            assert res['meta']['split_sha256']==base.sha256(base.split_path(base.CFG['small_run']['split_seed']))
        print(folder.name, 'saved metrics, row provenance and thresholds verified')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['refresh-metrics','embed','run-small-proto','check'])
    args=parser.parse_args()
    {'refresh-metrics':refresh_metrics,'embed':embed,'run-small-proto':small_proto,'check':check}[args.command]()
