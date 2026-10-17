"""Single entry point for the project. Subcommands are added stage by stage.
Run from the project root:  python run_experiment.py <command>
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_text(s):
    """Normalisation used ONLY for duplicate detection (never to alter the data)."""
    return " ".join(str(s).lower().split())


def load_data():
    d = CFG["dataset"]
    raw = ROOT / d["raw_dir"]
    train = pd.read_csv(raw / d["train_file"])
    test = pd.read_csv(raw / d["test_file"])
    cats = json.loads((raw / d["categories_file"]).read_text())
    return train, test, cats


def fingerprint():
    d = CFG["dataset"]; raw = ROOT / d["raw_dir"]
    return {k: sha256(raw / d[k]) for k in ("train_file", "test_file", "categories_file")}


def verify_data():
    train, test, cats = load_data()
    out = {"fingerprint_sha256": fingerprint()}
    out["columns"] = list(train.columns)
    out["n_train"], out["n_test"] = len(train), len(test)
    out["n_labels_json"] = len(cats)
    out["n_labels_train"] = train.category.nunique()
    out["n_labels_test"] = test.category.nunique()
    out["labels_match_json"] = (set(cats) == set(train.category) == set(test.category))
    out["null_text_or_label"] = int(train.isna().sum().sum() + test.isna().sum().sum())
    tc, ec = train.category.value_counts(), test.category.value_counts()
    out["train_per_class_min_med_max"] = [int(tc.min()), int(tc.median()), int(tc.max())]
    out["test_per_class_min_med_max"] = [int(ec.min()), int(ec.median()), int(ec.max())]
    out["test_per_class_unique_counts"] = sorted(set(int(x) for x in ec.values))

    for name, df in (("train", train), ("test", test)):
        df["norm"] = df.text.map(norm_text)
    # exact + normalised duplicates inside each split
    out["dup_exact_within_train"] = int(train.duplicated("text").sum())
    out["dup_norm_within_train"] = int(train.duplicated("norm").sum())
    out["dup_exact_within_test"] = int(test.duplicated("text").sum())
    out["dup_norm_within_test"] = int(test.duplicated("norm").sum())
    # conflicting labels: same normalised text, >1 distinct label
    for name, df in (("train", train), ("test", test)):
        g = df.groupby("norm").category.nunique()
        out[f"conflicting_label_texts_{name}"] = int((g > 1).sum())
    # train/test overlap (leakage risk)
    ov = set(train.norm) & set(test.norm)
    out["train_test_overlap_texts_norm"] = len(ov)
    both = pd.concat([train.assign(split="train"), test.assign(split="test")])
    g = both.groupby("norm").category.nunique()
    out["conflicting_label_texts_across_train_test"] = int((g > 1).sum())
    ex = both[both.norm.isin(g[g > 1].index)].sort_values("norm")[["split", "text", "category"]]
    out["conflict_examples"] = ex.head(10).to_dict("records")
    ex2 = train[train.duplicated("norm", keep=False)].sort_values("norm")[["text", "category"]]
    out["within_train_dup_examples"] = ex2.head(6).to_dict("records")
    out["text_len_words_train_mean_max"] = [round(train.text.str.split().str.len().mean(), 2),
                                            int(train.text.str.split().str.len().max())]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_verification.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if "examples" not in k}, indent=2))
    print("\nconflict examples:", json.dumps(out["conflict_examples"][:6], indent=1))
    print("within-train duplicate examples:", json.dumps(out["within_train_dup_examples"], indent=1))


def sanity_baseline():
    """Closed-set TF-IDF+LR on all 77 official classes. A sanity check of the pipeline,
    NOT part of the few-shot/unknown-intent study."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    import sklearn
    c = CFG["sanity_baseline"]
    train, test, _ = load_data()
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=tuple(c["tfidf"]["ngram_range"]),
                          sublinear_tf=c["tfidf"]["sublinear_tf"],
                          lowercase=c["tfidf"]["lowercase"], min_df=c["tfidf"]["min_df"])
    Xtr = vec.fit_transform(train.text)           # fit vocabulary on TRAIN only
    Xte = vec.transform(test.text)
    clf = LogisticRegression(C=c["logreg"]["C"], max_iter=c["logreg"]["max_iter"],
                             solver=c["logreg"]["solver"], random_state=c["random_state"])
    clf.fit(Xtr, train.category)
    pred = clf.predict(Xte)
    res = {"stage": "sanity_closed_set_77", "sklearn": sklearn.__version__,
           "config": c, "dataset_fingerprint": fingerprint(),
           "vocab_size": len(vec.vocabulary_),
           "accuracy": float(accuracy_score(test.category, pred)),
           "macro_f1": float(f1_score(test.category, pred, average="macro")),
           "fit_seconds": round(time.time() - t0, 2)}
    (ROOT / "results" / "sanity_baseline.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"text": test.text, "true": test.category, "pred": pred}).to_csv(
        ROOT / "results" / "sanity_baseline_predictions.csv", index=False)
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- Stage 2: splits
import inspect

SPLIT_DIR = ROOT / "results" / "splits"


def clean_text(s):
    """Model input: whitespace-collapsed text (raw CSV is never modified)."""
    return " ".join(str(s).split())


def prepare(df):
    df = df.copy()
    df["row"] = np.arange(len(df))          # positional row id in the official CSV
    df["clean"] = df.text.map(clean_text)
    df["key"] = df.clean.str.lower()        # duplicate-detection key
    return df


def filter_train(train, test):
    """Test-overlap integrity check. Official-train rows whose duplicate key (lowercase,
    whitespace-collapsed) occurs in the official test set are EXCLUDED from every learning,
    development and adaptation partition. Raw CSVs and the official test set are untouched;
    the `row` ids (positions in the raw train.csv) of kept rows are preserved.
    Uses no model output or test performance."""
    hit = train.key.isin(set(test.key))
    return train[~hit], train[hit]


def make_split(seed, S, train, test):
    """Category allocation + message splits for one category-split seed.
    All partitions are generated from the FILTERED training rows."""
    labels = sorted(train.category.unique())
    perm = [labels[i] for i in np.random.RandomState(seed).permutation(len(labels))]
    a, b, c = S["n_supported"], S["n_new"], S["n_dev_unknown"]
    sup, new = sorted(perm[:a]), sorted(perm[a:a + b])
    dev_unk, test_unk = sorted(perm[a + b:a + b + c]), sorted(perm[a + b + c:])
    assert len(sup) + len(new) + len(dev_unk) + len(test_unk) == len(labels) == 77
    role = {**{k: "supported" for k in sup}, **{k: "new" for k in new},
            **{k: "dev_unknown" for k in dev_unk}, **{k: "test_unknown" for k in test_unk}}
    train_f, excluded = filter_train(train, test)        # <-- BEFORE any partition
    rng = np.random.RandomState(1000 + seed)
    train_sup, dev_sup, dev_new, pools = [], [], [], {}
    for cls in sorted(sup + new):               # fixed order => reproducible draws
        g = train_f[train_f.category == cls]
        keys = sorted(g.key.unique())
        n_dev = max(1, int(round(S["dev_fraction"] * len(keys))))
        dev_keys = {keys[i] for i in rng.permutation(len(keys))[:n_dev]}
        is_dev = g.key.isin(dev_keys)           # duplicate groups stay on one side
        (dev_sup if cls in sup else dev_new).extend(g[is_dev].row.tolist())
        rest = g[~is_dev]
        if cls in sup:
            train_sup.extend(rest.row.tolist())
        else:                                   # adaptation pool: 1 row per distinct key
            pools[cls] = sorted(rest.drop_duplicates("key").row.tolist())
            if len(pools[cls]) < max(S["shots"]):
                raise ValueError(f"{cls}: only {len(pools[cls])} eligible messages")
    adapt = {}
    for s in S["adapt_seeds"]:
        r = np.random.RandomState(10000 + 100 * seed + s)
        adapt[str(s)] = {}
        for cls in new:
            order = r.permutation(len(pools[cls]))[:max(S["shots"])]
            adapt[str(s)][cls] = [int(pools[cls][i]) for i in order]   # prefix 1/5/10 => nested
    dev_unknown_rows = train_f[train_f.category.isin(dev_unk)].row.tolist()
    exc_rows = [{"row": int(r), "category": cat, "role": role[cat]}
                for r, cat in zip(excluded.row, excluded.category)]
    by_role = {k: sum(e["role"] == k for e in exc_rows)
               for k in ("supported", "new", "dev_unknown", "test_unknown")}
    return {"classes": {"supported": sup, "new": new, "dev_unknown": dev_unk, "test_unknown": test_unk},
            "excluded_test_overlap": {"n_rows": len(exc_rows), "rows": exc_rows, "n_by_role": by_role},
            "train_supported_rows": [int(x) for x in train_sup],
            "dev_supported_rows": [int(x) for x in dev_sup],
            "dev_new_rows": [int(x) for x in dev_new],
            "dev_unknown_rows": [int(x) for x in dev_unknown_rows],
            "adapt_pool_sizes": {k: len(v) for k, v in pools.items()},
            "adapt": adapt}


def split_code_hash():
    s = "".join(inspect.getsource(f) for f in (clean_text, prepare, filter_train, make_split))
    return hashlib.sha256(s.encode()).hexdigest()


def split_meta(seed):
    return {"split_seed": seed, "config": CFG["splits"], "dataset_fingerprint": fingerprint(),
            "code_hash": split_code_hash()}


def load_prepared():
    train, test, _ = load_data()
    return prepare(train), prepare(test)


def split_path(seed):
    return SPLIT_DIR / f"split_seed{seed}.json"


def load_split(seed):
    """Load a split file ONLY if it matches current config, data and code."""
    p = split_path(seed)
    d = json.loads(p.read_text())
    if d["meta"] != split_meta(seed):
        raise RuntimeError(f"{p} is stale (config/data/code changed). Re-run make-splits.")
    return d


def make_splits():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_prepared()
    _, exc = filter_train(train, test)
    (ROOT / "results" / "test_overlap_exclusion.json").write_text(json.dumps({
        "policy": CFG["splits"]["overlap_policy"], "dataset_fingerprint": fingerprint(),
        "n_rows_excluded": len(exc), "n_distinct_keys": int(exc.key.nunique()),
        "by_category": exc.category.value_counts().sort_index().to_dict(),
        "rows": [{"row": int(r), "category": c, "text": t} for r, c, t in zip(exc.row, exc.category, exc.clean)]},
        indent=1))
    print(f"test-overlap exclusion: {len(exc)} train rows excluded ({exc.key.nunique()} keys)")
    for seed in CFG["splits"]["split_seeds"]:
        p = split_path(seed)
        if p.exists() and json.loads(p.read_text()).get("meta") == split_meta(seed):
            print(f"seed {seed}: valid file exists, reused ({p.name})"); continue
        d = make_split(seed, CFG["splits"], train, test)
        d["meta"] = split_meta(seed)
        p.write_text(json.dumps(d))
        print(f"seed {seed}: written {p.name}")


def check_splits():
    """Focused boundary checks (Stage 2 checks kept; test-overlap, full-regeneration,
    seeds/shots and pool-size checks added)."""
    train, test = load_prepared()
    S = CFG["splits"]
    key = train.set_index("row").key
    cat = train.set_index("row").category
    test_keys = set(test.key)
    excl_rows = set(train.loc[train.key.isin(test_keys), "row"])      # independent recomputation
    for seed in S["split_seeds"]:
        d = load_split(seed)
        C = {k: set(v) for k, v in d["classes"].items()}
        # 1. category separation
        sizes = [len(C[k]) for k in ("supported", "new", "dev_unknown", "test_unknown")]
        assert sizes == [S["n_supported"], S["n_new"], S["n_dev_unknown"], S["n_test_unknown"]], sizes
        allc = set().union(*C.values()); assert len(allc) == 77 == sum(sizes)
        # 2. message separation (by row AND by duplicate key)
        tr, dv, dn, du = (set(d[k]) for k in ("train_supported_rows", "dev_supported_rows",
                                              "dev_new_rows", "dev_unknown_rows"))
        assert not tr & dv and not tr & dn and not dv & dn
        assert {cat[r] for r in tr | dv} <= C["supported"]
        assert {cat[r] for r in dn} == C["new"] and {cat[r] for r in du} == C["dev_unknown"]
        ktr = {key[r] for r in tr}
        assert not ktr & {key[r] for r in dv}, "dup key straddles supported train/dev"
        # 3. final-test-unknown classes never appear in any fitting/dev/adaptation set
        adapt_rows = {r for s_ in d["adapt"].values() for rows in s_.values() for r in rows}
        used = tr | dv | dn | du | adapt_rows
        assert not {cat[r] for r in used} & C["test_unknown"], "test-unknown class leaked"
        # 4. adaptation: counts, disjointness, nesting
        devkeys_new = {key[r] for r in dn}
        prev = None
        for s_, per_cls in d["adapt"].items():
            assert set(per_cls) == C["new"]
            for cls, rows in per_cls.items():
                assert len(rows) == max(S["shots"]) == len(set(rows)) == len({key[r] for r in rows})
                assert {cat[r] for r in rows} == {cls}
                assert not {key[r] for r in rows} & devkeys_new, "adaptation/dev overlap"
                for k1, k2 in zip(S["shots"], S["shots"][1:]):
                    assert set(rows[:k1]) <= set(rows[:k2])      # nested by construction
            if prev is not None:
                assert per_cls != prev, "different adapt seeds gave identical samples"
            prev = per_cls
        # 5. NEW: no train/dev/dev-unknown/adaptation text key overlaps ANY official-test key
        assert not {key[r] for r in used} & test_keys, "partition text key overlaps official test"
        assert not used & excl_rows, "an excluded row id is used in a partition"
        # 5b. NEW: recorded exclusions equal the independent recomputation
        rec = d["excluded_test_overlap"]
        assert {e["row"] for e in rec["rows"]} == excl_rows and rec["n_rows"] == len(excl_rows)
        assert all(e["category"] == cat[e["row"]] for e in rec["rows"])
        # 5c. NEW: partitions are complete w.r.t. filtered rows (nothing silently dropped)
        keep = train[~train.row.isin(excl_rows)]
        assert tr | dv == set(keep[keep.category.isin(C["supported"])].row), "supported rows not accounted for"
        assert du == set(keep[keep.category.isin(C["dev_unknown"])].row), "dev-unknown rows not accounted for"
        # 6. NEW: configured adaptation seeds and every requested shot count present
        assert sorted(d["adapt"]) == sorted(str(x) for x in S["adapt_seeds"]), "adapt seeds mismatch"
        for s_, per_cls in d["adapt"].items():
            for cls, rows in per_cls.items():
                for k in S["shots"]:
                    assert len(rows[:k]) == k == len(set(rows[:k])), (s_, cls, k)
        # 7. NEW: enough distinct eligible messages per new class for the largest shot count
        for cls in C["new"]:
            g = keep[keep.category == cls]
            elig = set(g.key) - devkeys_new
            assert len(elig) >= max(S["shots"]), (cls, len(elig))
            assert d["adapt_pool_sizes"][cls] == len(elig), "saved pool size != independent recomputation"
        # 8. NEW: regeneration reproduces EVERY saved field (meta excluded; meta is checked by load_split)
        regen = json.loads(json.dumps(make_split(seed, S, train, test)))
        saved = {k: v for k, v in d.items() if k != "meta"}
        assert set(regen) == set(saved), (set(regen) ^ set(saved))
        bad = [k for k in regen if regen[k] != saved[k]]
        assert not bad, f"regeneration differs from saved file in fields: {bad}"
        n_ts = int(test.category.isin(C["supported"]).sum()); n_tn = int(test.category.isin(C["new"]).sum())
        n_tu = int(test.category.isin(C["test_unknown"]).sum())
        print(f"seed {seed}: OK | train_sup={len(tr)} dev_sup={len(dv)} dev_new={len(dn)} dev_unk={len(du)} "
              f"| excluded train rows={len(excl_rows)} by role={rec['n_by_role']} "
              f"| min adapt pool={min(d['adapt_pool_sizes'].values())} "
              f"| test: orig={n_ts} new={n_tn} test_unknown={n_tu}")
    print("all split checks passed (executed)")


# ---------------------------------------------------------------- Stage 3: TF-IDF+LR small run
SMALL_DIR = ROOT / "results" / "small_run"


def fit_tfidf_lr(texts, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    c = CFG["few_shot_tfidf_lr"]
    vec = TfidfVectorizer(ngram_range=tuple(c["tfidf"]["ngram_range"]), sublinear_tf=c["tfidf"]["sublinear_tf"],
                          lowercase=c["tfidf"]["lowercase"], min_df=c["tfidf"]["min_df"])
    X = vec.fit_transform(texts)                     # vocabulary fitted on the training texts only
    clf = LogisticRegression(C=c["logreg"]["C"], max_iter=c["logreg"]["max_iter"],
                             solver=c["logreg"]["solver"], random_state=c["random_state"])
    clf.fit(X, labels)
    return vec, clf


def lr_predict(model, texts):
    """Returns (predicted class, acceptance score = max predicted class probability)."""
    vec, clf = model
    P = clf.predict_proba(vec.transform(list(texts)))
    return clf.classes_[P.argmax(1)], P.max(1)


def pick_threshold(supported_dev_scores):
    """Accept iff score >= t, with t the lower 10th percentile of SUPPORTED-dev scores
    (numpy 'lower' method => t is an observed score, so acceptance >= ~90% on that dev set)."""
    T = CFG["threshold"]
    return float(np.quantile(np.asarray(supported_dev_scores), T["quantile"], method=T["quantile_method"]))


def rows_hash(rows):
    return hashlib.sha256(json.dumps(sorted(int(r) for r in rows)).encode()).hexdigest()[:16]


def stage3_code_hash():
    s = "".join(inspect.getsource(f) for f in (fit_tfidf_lr, lr_predict, pick_threshold, run_small_compute))
    return hashlib.sha256(s.encode()).hexdigest()


def run_small_compute(seed, adapt_seed, shots):
    train, test = load_prepared()
    d = load_split(seed)
    txt, cat = train.set_index("row").clean, train.set_index("row").category
    C = d["classes"]; orig_c, new_c = C["supported"], C["new"]
    tr, dv, dn, du = (d[k] for k in ("train_supported_rows", "dev_supported_rows", "dev_new_rows", "dev_unknown_rows"))
    ad = [r for cls in new_c for r in d["adapt"][str(adapt_seed)][cls][:shots]]
    assert len(ad) == len(new_c) * shots
    T = CFG["threshold"]
    dev_scores, prov = [], {}
    # ---- before adaptation: 45 supported classes
    t0 = time.time(); m0 = fit_tfidf_lr(txt[tr], cat[tr]); fit0 = time.time() - t0
    _, s_dsup0 = lr_predict(m0, txt[dv]); _, s_dunk0 = lr_predict(m0, txt[du])
    thr0 = pick_threshold(s_dsup0)
    prov["pre"] = {"threshold": thr0, "calibration_population": "dev_supported", "n_calibration": len(dv),
                   "calibration_rows_hash": rows_hash(dv), "dev_unknown_rows_hash": rows_hash(du),
                   "uses_official_test": False}
    # ---- after adaptation: refit on original training + adaptation examples (vectorizer refit too)
    t0 = time.time(); m1 = fit_tfidf_lr(pd.concat([txt[tr], txt[ad]]), pd.concat([cat[tr], cat[ad]]))
    adapt_seconds = time.time() - t0
    _, s_dsup1 = lr_predict(m1, txt[dv + dn]); _, s_dunk1 = lr_predict(m1, txt[du])
    thr1 = pick_threshold(s_dsup1)
    prov["post"] = {"threshold": thr1, "calibration_population": "dev_supported + dev_new",
                    "n_calibration": len(dv) + len(dn), "calibration_rows_hash": rows_hash(dv + dn),
                    "dev_unknown_rows_hash": rows_hash(du), "uses_official_test": False}
    for stage, ss, su, thr in (("pre", s_dsup0, s_dunk0, thr0), ("post", s_dsup1, s_dunk1, thr1)):
        prov[stage]["achieved_dev_supported_acceptance"] = float((ss >= thr).mean())
        prov[stage]["dev_unknown_false_acceptance"] = float((su >= thr).mean())
    dev_scores = (
        [{"stage": "pre", "population": "dev_supported", "row": r, "score": float(x)} for r, x in zip(dv, s_dsup0)] +
        [{"stage": "pre", "population": "dev_unknown", "row": r, "score": float(x)} for r, x in zip(du, s_dunk0)] +
        [{"stage": "post", "population": "dev_supported", "row": r, "score": float(x)} for r, x in zip(dv, s_dsup1[:len(dv)])] +
        [{"stage": "post", "population": "dev_new", "row": r, "score": float(x)} for r, x in zip(dn, s_dsup1[len(dv):])] +
        [{"stage": "post", "population": "dev_unknown", "row": r, "score": float(x)} for r, x in zip(du, s_dunk1)])
    # ---- official test, three separate populations
    pops = {"orig_supported": test[test.category.isin(orig_c)], "new": test[test.category.isin(new_c)],
            "test_unknown": test[test.category.isin(C["test_unknown"])]}
    preds = []
    for stage, m, thr in (("pre", m0, thr0), ("post", m1, thr1)):
        for pop, df in pops.items():
            pr, sc = lr_predict(m, df.clean)
            preds.append(pd.DataFrame({"stage": stage, "population": pop, "row": df.row.values,
                                       "text": df.clean.values, "true": df.category.values, "pred": pr,
                                       "score": sc, "accepted": sc >= thr, "threshold": thr}))
    preds = pd.concat(preds, ignore_index=True)
    return {"preds": preds, "dev_scores": pd.DataFrame(dev_scores), "prov": prov, "adapt_seconds": adapt_seconds,
            "fit_seconds_pre": fit0, "n_adapt": len(ad), "classes": C}


def macro_f1_manual(true, pred, labels):
    f = []
    for l in labels:
        tp = np.sum((true == l) & (pred == l)); fp = np.sum((true != l) & (pred == l)); fn = np.sum((true == l) & (pred != l))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def metrics_from_predictions(preds, C):
    """Plain numpy/pandas metric definitions (also used to re-verify saved files)."""
    out = {}
    for stage in ("pre", "post"):
        P = preds[preds.stage == stage]; m = {}
        o, n, u = (P[P.population == k] for k in ("orig_supported", "new", "test_unknown"))
        t_, p_ = o.true.values, o.pred.values
        m["closed_set_macro_f1_original_intents"] = macro_f1_manual(t_, p_, C["supported"])
        m["closed_set_accuracy_original_intents"] = float((t_ == p_).mean())
        m["orig_supported_acceptance_rate"] = float(o.accepted.mean())
        m["orig_correct_routing_rate"] = float(((o.true == o.pred) & o.accepted).mean())
        m["orig_accuracy_among_accepted"] = float((o.true == o.pred)[o.accepted].mean()) if o.accepted.any() else None
        m["test_unknown_false_acceptance"] = float(u.accepted.mean())
        if stage == "pre":     # new classes are still UNSUPPORTED here: accepting them is a false acceptance
            m["new_as_unsupported_false_acceptance"] = float(n.accepted.mean())
        else:
            m["closed_set_macro_f1_new_intents"] = macro_f1_manual(n.true.values, n.pred.values, C["new"])
            m["closed_set_accuracy_new_intents"] = float((n.true == n.pred).mean())
            m["new_acceptance_rate"] = float(n.accepted.mean())
            m["new_correct_routing_rate"] = float(((n.true == n.pred) & n.accepted).mean())
            m["new_accuracy_among_accepted"] = float((n.true == n.pred)[n.accepted].mean()) if n.accepted.any() else None
            S = pd.concat([o, n])      # all supported requests after addition
            m["all_supported_acceptance_rate"] = float(S.accepted.mean())
            m["all_supported_correct_routing_rate"] = float(((S.true == S.pred) & S.accepted).mean())
            m["all_supported_accuracy_among_accepted"] = float((S.true == S.pred)[S.accepted].mean())
        out[stage] = m
    pre, post = out["pre"], out["post"]
    out["change_after_addition"] = {
        "original_macro_f1": post["closed_set_macro_f1_original_intents"] - pre["closed_set_macro_f1_original_intents"],
        "original_acceptance_rate": post["orig_supported_acceptance_rate"] - pre["orig_supported_acceptance_rate"],
        "original_correct_routing_rate": post["orig_correct_routing_rate"] - pre["orig_correct_routing_rate"],
        "test_unknown_false_acceptance": post["test_unknown_false_acceptance"] - pre["test_unknown_false_acceptance"]}
    return out


def small_meta(seed, adapt_seed, shots):
    split_file = split_path(seed)
    return {"split_seed": seed, "adapt_seed": adapt_seed, "shots": shots,
            "few_shot_tfidf_lr": CFG["few_shot_tfidf_lr"], "threshold_cfg": CFG["threshold"],
            "dataset_fingerprint": fingerprint(), "split_meta": json.loads(split_file.read_text())["meta"],
            "split_file_sha256": sha256(split_file), "stage3_code_hash": stage3_code_hash()}


def run_small(force=False):
    c = CFG["small_run"]; seed, a_seed, shots = c["split_seed"], c["adapt_seed"], c["shots"]
    SMALL_DIR.mkdir(parents=True, exist_ok=True)
    mp = SMALL_DIR / "metrics.json"
    meta = small_meta(seed, a_seed, shots)
    if mp.exists() and not force and json.loads(mp.read_text()).get("meta") == meta:
        print("valid small-run outputs exist (meta matches); reused. Use --force to recompute.")
    else:
        r = run_small_compute(seed, a_seed, shots)
        r["preds"].to_csv(SMALL_DIR / "predictions.csv", index=False)
        r["dev_scores"].to_csv(SMALL_DIR / "dev_scores.csv", index=False)
        res = {"meta": meta, "stage": "small_run_tfidf_lr", "threshold_provenance": r["prov"],
               "adaptation_seconds_refit_tfidf_and_lr": round(r["adapt_seconds"], 3),
               "initial_fit_seconds": round(r["fit_seconds_pre"], 3), "n_adaptation_examples": r["n_adapt"],
               "metrics": metrics_from_predictions(r["preds"], r["classes"])}
        mp.write_text(json.dumps(res, indent=2))
    check_small_run()
    print(json.dumps({k: v for k, v in json.loads(mp.read_text()).items() if k != "meta"}, indent=2))


def check_small_run():
    """Re-verify saved small-run outputs: provenance, thresholds, counts and metrics."""
    c = CFG["small_run"]; seed, a_seed, shots = c["split_seed"], c["adapt_seed"], c["shots"]
    res = json.loads((SMALL_DIR / "metrics.json").read_text())
    assert res["meta"] == small_meta(seed, a_seed, shots), "small-run outputs are stale"
    d = load_split(seed); C = d["classes"]
    train, test = load_prepared()
    preds = pd.read_csv(SMALL_DIR / "predictions.csv"); dev = pd.read_csv(SMALL_DIR / "dev_scores.csv")
    T = CFG["threshold"]
    assert res["n_adaptation_examples"] == len(C["new"]) * shots
    assert set(preds[preds.stage == "pre"].population) == {"orig_supported", "new", "test_unknown"}
    # threshold provenance: recomputed from saved DEV scores only; dev rows disjoint from test by key
    tk = set(test.key); dkey = train.set_index("row").key
    assert not {dkey[r] for r in dev.row} & tk, "dev rows overlap official test"
    for stage, pops in (("pre", ["dev_supported"]), ("post", ["dev_supported", "dev_new"])):
        D = dev[dev.stage == stage]
        sc = D[D.population.isin(pops)].score.values
        thr = float(np.quantile(sc, T["quantile"], method=T["quantile_method"]))
        p = res["threshold_provenance"][stage]
        assert abs(thr - p["threshold"]) < 1e-12 and p["uses_official_test"] is False
        assert abs(float((sc >= thr).mean()) - p["achieved_dev_supported_acceptance"]) < 1e-12
        assert p["n_calibration"] == len(sc)
        assert (preds[preds.stage == stage].threshold == thr).all(), "test predictions used another threshold"
        un = D[D.population == "dev_unknown"].score.values
        assert abs(float((un >= thr).mean()) - p["dev_unknown_false_acceptance"]) < 1e-12
    assert set(preds[preds.stage == "post"].accepted) <= {True, False}
    # populations come from the right classes, and test-unknown classes are the configured ones
    for pop, cls in (("orig_supported", C["supported"]), ("new", C["new"]), ("test_unknown", C["test_unknown"])):
        assert set(preds[preds.population == pop].true) == set(cls)
    # acceptance is exactly score >= threshold
    assert ((preds.score >= preds.threshold) == preds.accepted).all()
    # metrics recomputed from the saved predictions
    again = metrics_from_predictions(preds, C)
    assert json.dumps(again, sort_keys=True) == json.dumps(res["metrics"], sort_keys=True) or \
        all(abs(again[s][k] - res["metrics"][s][k]) < 1e-12 for s in ("pre", "post") for k in again[s]
            if again[s][k] is not None), "saved metrics differ from recomputation"
    print("small-run checks passed (executed): provenance, thresholds, counts, metric recomputation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    cmds = {"verify-data": verify_data, "sanity-baseline": sanity_baseline,
            "make-splits": make_splits, "check-splits": check_splits,
            "run-small": run_small, "check-small-run": check_small_run}
    ap.add_argument("command", choices=list(cmds))
    ap.add_argument("--force", action="store_true", help="run-small: recompute even if valid outputs exist")
    args = ap.parse_args()
    cmds[args.command](force=args.force) if args.command == "run-small" else cmds[args.command]()
