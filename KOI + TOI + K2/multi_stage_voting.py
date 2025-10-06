#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_stage_voting.py
---------------------
Pipeline de votação multi-domínios (KOI, TOI, K2).
Step 2: constrói votante local calibrado por domínio (RF + XGB + Stack -> soft vote + Isotonic)
Step 3: meta-voto multi-domínios (soft vote ponderado com shrinkage+cap OU LR regularizada)
Infer: aplica KOI/TOI/K2 (o que existir no CSV) e combina no voto final.
"""
import os, json, argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

RND = 42

# =========================
# Feature sets por domínio
# =========================
KOI_FEATS = ["koi_period","koi_duration","koi_depth","koi_prad","koi_insol","koi_teq"]
TOI_FEATS = ["pl_orbper","pl_trandurh","pl_trandep","pl_rade","pl_insol","pl_eqt","st_teff","st_logg"]
K2_FEATS  = ["pl_orbper","pl_trandur","pl_trandep","pl_rade","pl_insol","pl_eqt","st_teff","st_logg"]  # duração conforme seu pré-proc

DOMAIN_FEATURES = {
    "KOI": KOI_FEATS,
    "TOI": TOI_FEATS,
    "K2" : K2_FEATS,
}

# =========================
# Utilitários
# =========================
def ensure(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    Xtr = data["X_train"]; Xte = data["X_test"]
    ytr = data["y_train"]; yte = data["y_test"]
    feats = data["feature_names"].tolist() if "feature_names" in data else None
    return Xtr, Xte, ytr, yte, feats

def soft_vote_proba(models: List, X, weights: np.ndarray = None):
    ps = []
    for m in models:
        if hasattr(m, "predict_proba"):
            ps.append(m.predict_proba(X)[:,1])
        elif hasattr(m, "decision_function"):
            df = m.decision_function(X)
            p = 1/(1+np.exp(-df))
            ps.append(p)
        else:
            ps.append(m.predict(X).astype(float))
    P = np.vstack(ps).T  # (n, k)
    if weights is None:
        w = np.ones(P.shape[1], dtype=float) / P.shape[1]
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (w.sum() if w.sum() > 0 else 1.0)
    out = (P * w).sum(axis=1)
    return np.clip(out, 1e-6, 1-1e-6)

def fit_isotonic(p_raw: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y)
    return iso

def evaluate(y, p, thr=0.5, prefix=""):
    yhat = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    auroc = roc_auc_score(y, p) if len(np.unique(y))==2 else np.nan
    return {
        f"{prefix}f1": float(f1),
        f"{prefix}precision": float(prec),
        f"{prefix}recall": float(rec),
        f"{prefix}auroc": float(auroc),
        f"{prefix}thr": float(thr)
    }

def temp_scale(p, T=1.0):
    p = np.clip(p, 1e-6, 1-1e-6)
    logit = np.log(p/(1-p))
    return 1/(1+np.exp(-logit/float(T)))

# =========================
# STEP 2: votante local
# =========================
def build_local_voter(run_dir: Path, domain_tag: str):
    """
    run_dir contém:
      preprocessed/{data_ready.npz, preprocessor.pkl, vt.pkl}
      models_cpu/{rf_model.pkl, xgb_model.pkl, stack_model.pkl}
    Saída:
      voter/{models.pkl, calibrator_isotonic.pkl, voter_meta.json, metrics.json}
    """
    models_dir = Path(run_dir)/"models_cpu"
    prep_dir   = Path(run_dir)/"preprocessed"
    out_dir    = Path(run_dir)/"voter"
    ensure(out_dir)

    # carrega modelos (nomes padrão; ignore se algum faltar)
    models = {}
    for name in ["rf_model.pkl","xgb_model.pkl","stack_model.pkl"]:
        p = models_dir/name
        if p.exists():
            models[name.split("_")[0]] = joblib.load(p)
    if not models:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {models_dir}")

    # split
    Xtr, Xte, ytr, yte, feat_names = load_npz(prep_dir/"data_ready.npz")

    # previsões individuais
    indiv = {}
    for key, mdl in models.items():
        if hasattr(mdl, "predict_proba"):
            indiv[key] = mdl.predict_proba(Xte)[:,1]
        else:
            indiv[key] = mdl.decision_function(Xte)

    # pesos por F1 nos membros
    f_scores = {k: f1_score(yte, (p>=0.5).astype(int)) for k,p in indiv.items()}
    weights = np.array([max(f_scores.get("rf",1e-6),1e-6),
                        max(f_scores.get("xgb",1e-6),1e-6),
                        max(f_scores.get("stack",1e-6),1e-6)], dtype=float)

    # soft vote bruto
    ordered_models = [models[k] for k in ["rf","xgb","stack"] if k in models]
    used_weights = weights[:len(ordered_models)]
    p_vote_raw = soft_vote_proba(ordered_models, Xte, weights=used_weights)

    # calibração isotônica
    iso = fit_isotonic(p_vote_raw, yte)
    p_vote_cal = iso.transform(p_vote_raw)

    # threshold ótimo (grid)
    grid = np.linspace(0.1,0.9,33)
    fgrid = [f1_score(yte, (p_vote_cal>=t).astype(int)) for t in grid]
    thr = float(grid[int(np.argmax(fgrid))])

    # salvar
    joblib.dump(models, out_dir/"models.pkl")
    joblib.dump(iso, out_dir/"calibrator_isotonic.pkl")
    voter_meta = {"domain": domain_tag, "weights": used_weights.tolist(), "threshold": thr}
    with open(out_dir/"voter_meta.json","w",encoding="utf-8") as f:
        json.dump(voter_meta, f, indent=2, ensure_ascii=False)

    # métricas
    metrics = {}
    for k, p in indiv.items():
        metrics.update(evaluate(yte, p, prefix=f"{k}_"))
    metrics.update(evaluate(yte, p_vote_cal, thr=thr, prefix="vote_"))
    with open(out_dir/"metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[{domain_tag}] Step 2 pronto em: {out_dir}")
    print("Pesos (ordem dos modelos salvos):", voter_meta["weights"])
    print("Threshold ótimo (F1):", thr)

# =========================
# STEP 3: meta-voto multi
# =========================
def build_step3_multi(
    runs,                     # dict com { "koi": Path("koi_run"), "toi": Path("toi_run"), "k2": Path("k2_run"), ... }
    out_dir,                  # Path destino para salvar os artefatos do Step 3
    alpha=0.5,                # fator para ponderar pesos por desempenho (ex.: F1)
    cap_ratio=1.75,           # limite de desequilíbrio entre pesos
    use_logreg=False,         # se True, usa LogisticRegression como meta-votante
    random_state=42           # semente
):
    """
    Constrói o meta-voto (Step 3) a partir das saídas calibradas dos domínios (Step 2).
    Salva:
      - final_vote_meta.pkl (contendo threshold aprendido, domínios, pesos, e flag use_logreg)
      - opcionalmente final_vote_lr.pkl (modelo LR usado como meta-votante)
    Espera encontrar em cada 'run' arquivos com:
      - 'calib_probs.npy' (probabilidades calibradas de validação do domínio)
      - 'y_valid.npy'      (rótulos 0/1 correspondentes)
      - 'domain_perf.json' (métricas de validação; ao menos 'f1' ou 'roc_auc')
    """
    import json
    from pathlib import Path
    import numpy as np
    import joblib
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(random_state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1) Carrega material de cada domínio ===
    per_domain = []  # cada item: {"name", "p_cal", "y", "f1", "auc"}
    for name, rdir in runs.items():
        rdir = Path(rdir)
        p_cal_path = rdir / "calib_probs.npy"
        y_path     = rdir / "y_valid.npy"
        perf_path  = rdir / "domain_perf.json"

        if not (p_cal_path.exists() and y_path.exists() and perf_path.exists()):
            print(f"[WARN] Step 3 ignorando domínio '{name}' (arquivos ausentes).")
            continue

        p_cal = np.load(p_cal_path)        # shape (N,)
        y     = np.load(y_path).astype(int)

        with open(perf_path, "r") as f:
            perf = json.load(f)
        f1  = float(perf.get("f1", 0.0))
        auc = float(perf.get("roc_auc", perf.get("auc", 0.0)))

        # sanity
        if p_cal.ndim != 1 or p_cal.shape[0] != y.shape[0]:
            print(f"[WARN] Dimensões inconsistentes em '{name}', pulando.")
            continue

        per_domain.append({"name": name, "p_cal": p_cal, "y": y, "f1": f1, "auc": auc})

    if len(per_domain) < 2:
        raise RuntimeError("Step 3 requer >=2 domínios válidos com calibração/validação disponíveis.")

    # === 2) Deriva pesos por desempenho (simples e robusto) ===
    # peso base = (alpha * f1 + (1-alpha) * auc), normalizado e com cap de razão
    raw = np.array([alpha * d["f1"] + (1 - alpha) * d["auc"] for d in per_domain], dtype=float)
    # evita todos zeros
    if np.all(raw <= 0):
        raw = np.ones_like(raw)
    raw = np.maximum(raw, 1e-6)

    # normaliza
    w = raw / raw.sum()

    # aplica cap de desequilíbrio (máx/min <= cap_ratio)
    w_sorted_idx = np.argsort(w)
    w_min = w[w_sorted_idx[0]]
    w_max = w[w_sorted_idx[-1]]
    if w_max / max(w_min, 1e-9) > cap_ratio:
        # "puxa" o maior para o limite e re-normaliza
        target_max = cap_ratio * w_min
        scale = target_max / w_max
        w[w_sorted_idx[-1]] = w[w_sorted_idx[-1]] * scale
        w = w / w.sum()

    # === 3) Monta matriz Z (probabilidades calibradas) e vetor y combinados ===
    # Usamos a concatenação simples (pilha) para calibrar o limiar no conjunto combinado.
    Z_list = [d["p_cal"] for d in per_domain]
    y_list = [d["y"] for d in per_domain]
    Z_cal = np.column_stack(Z_list)  # shape (N_total, n_domains) após concat por empilhamento de vetores
    y_cal = np.concatenate(y_list)

    # === 4) Aprende o meta-voto ===
    meta = {
        "domains": [d["name"] for d in per_domain],
        "weights": {d["name"]: float(wi) for d, wi in zip(per_domain, w)},
        "use_logreg": bool(use_logreg),
        "alpha": float(alpha),
        "cap_ratio": float(cap_ratio),
        "random_state": int(random_state),
    }

    if not use_logreg:
        # Média ponderada das probabilidades
        w_vec = w / (w.sum() if w.sum() > 0 else 1.0)
        p_meta_cal = np.dot(Z_cal, w_vec)

        # Busca de limiar ótimo (F1) no meta
        grid = np.linspace(0.1, 0.9, 33)  # passo ~0.025
        fgrid = [f1_score(y_cal, (p_meta_cal >= t).astype(int)) for t in grid]
        thr_final = float(grid[int(np.argmax(fgrid))])

        meta["threshold"] = thr_final
        joblib.dump(meta, out_dir / "final_vote_meta.pkl")
        print(f"[OK] Step 3 (média ponderada) salvo em {out_dir/'final_vote_meta.pkl'}  | thr={thr_final:.3f}")
    else:
        # Logistic Regression como meta-votante
        lr = LogisticRegression(max_iter=1000, C=0.5, random_state=random_state)
        lr.fit(Z_cal, y_cal)
        joblib.dump(lr, out_dir / "final_vote_lr.pkl")

        # (opcional) ainda otimiza limiar sobre a saída da LR
        p_lr = lr.predict_proba(Z_cal)[:, 1]
        grid = np.linspace(0.1, 0.9, 33)
        fgrid = [f1_score(y_cal, (p_lr >= t).astype(int)) for t in grid]
        thr_final = float(grid[int(np.argmax(fgrid))])

        meta["threshold"] = thr_final
        joblib.dump(meta, out_dir / "final_vote_meta.pkl")
        print(f"[OK] Step 3 (LogReg meta) salvo em {out_dir/'final_vote_meta.pkl'} e {out_dir/'final_vote_lr.pkl'} | thr={thr_final:.3f}")

    return meta

# =========================
# INFERÊNCIA
# =========================
def _load_domain_artifacts(run_dir: Path):
    pre  = joblib.load(Path(run_dir)/"preprocessed"/"preprocessor.pkl")
    vt   = joblib.load(Path(run_dir)/"preprocessed"/"vt.pkl")
    mdl  = joblib.load(Path(run_dir)/"voter"/"models.pkl")
    iso  = joblib.load(Path(run_dir)/"voter"/"calibrator_isotonic.pkl")
    meta = json.load(open(Path(run_dir)/"voter"/"voter_meta.json","r",encoding="utf-8"))
    return pre, vt, mdl, iso, meta

def infer(domains: List[Path], final_dir: Path, csv_in: Path, csv_out: Path):
    df = pd.read_csv(csv_in)
    out = pd.DataFrame(index=df.index)
    per_prob = {}

    for ddir in domains:
        name = Path(ddir).name
        if "koi" in name.lower():   dom_key = "KOI"
        elif "toi" in name.lower(): dom_key = "TOI"
        elif "k2" in name.lower():  dom_key = "K2"
        else:
            if set(KOI_FEATS).issubset(df.columns): dom_key = "KOI"
            elif set(TOI_FEATS).issubset(df.columns): dom_key = "TOI"
            elif set(K2_FEATS).issubset(df.columns): dom_key = "K2"
            else:
                print(f"[WARN] Não reconheci o domínio para {name}; pulando.")
                continue

        feats = DOMAIN_FEATURES[dom_key]
        if not set(feats).issubset(df.columns):
            out[f"p_{dom_key.lower()}"] = np.nan
            out[f"pred_{dom_key.lower()}"] = np.nan
            continue

        pre, vt, mdls, iso, meta = _load_domain_artifacts(ddir)
        X = df[feats].copy()
        X = pre.transform(X)
        X = vt.transform(X)

        ordered_models = [mdls[k] for k in ["rf","xgb","stack"] if k in mdls]
        w = np.array(meta["weights"], dtype=float)[:len(ordered_models)]
        p_raw = soft_vote_proba(ordered_models, X, w)
        p_cal = iso.transform(p_raw)
        thr = float(meta.get("threshold", 0.5))
        out[f"p_{dom_key.lower()}"] = p_cal
        out[f"pred_{dom_key.lower()}"] = (p_cal >= thr).astype(int)
        per_prob[dom_key] = p_cal

    # combinação final
    meta_final = joblib.load(Path(final_dir)/"final_vote_meta.pkl")
    domains_order = meta_final["domains"]
    Z = []
    for dname in domains_order:
        key = "KOI" if "koi" in dname.lower() else ("TOI" if "toi" in dname.lower() else ("K2" if "k2" in dname.lower() else dname))
        p = per_prob.get(key, None)
        Z.append(p if p is not None else np.full(len(df), 0.5))
    Z = np.column_stack(Z)

    if meta_final.get("use_logreg", False):
        lr = joblib.load(Path(final_dir)/"final_vote_lr.pkl")
        p_final = lr.predict_proba(Z)[:,1]
    else:
        weights = meta_final["weights"]
        w = np.array([weights[d] for d in domains_order], dtype=float)
        w = w / (w.sum() if w.sum() > 0 else 1.0)
        p_final = np.dot(Z, w)

    out["p_final"] = np.clip(p_final, 1e-6, 1-1e-6)
    thr_final = float(meta_final.get("threshold", 0.5))
    out["pred_final"] = (out["p_final"] >= thr_final).astype(int)
    out.to_csv(csv_out, index=False)
    print("Inferência salva em:", csv_out)

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Votação multi-domínios (KOI/TOI/K2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp2 = sub.add_parser("step2", help="Cria votante local calibrado para um domínio")
    sp2.add_argument("--run_dir", required=True, help="Diretório do domínio (contém models_cpu/ e preprocessed/)")
    sp2.add_argument("--domain", required=True, choices=["KOI","TOI","K2"], help="Tag do domínio (apenas para logging)")

    sp3 = sub.add_parser("step3", help="Cria meta-voto final (2+ domínios)")
    sp3.add_argument("--domains", nargs="+", required=True, help="Lista de diretórios de domínio (koi_run toi_run k2_run ...)")
    sp3.add_argument("--out_dir", default="final_vote", help="Diretório de saída do meta-voto")
    sp3.add_argument("--mode", default="soft-temp", choices=["soft","soft-temp"], help="Combinação: soft ou soft+temperature para domínios pequenos")
    sp3.add_argument("--alpha", type=float, default=0.5, help="Expoente do tamanho efetivo n^alpha nos pesos")
    sp3.add_argument("--cap_ratio", type=float, default=1.75, help="Limite para max(w)/min(w)")
    sp3.add_argument("--small_n", type=int, default=400, help="Abaixo deste n usa temperature scaling")
    sp3.add_argument("--temp_small", type=float, default=1.5, help="Temperatura para domínios pequenos")
    sp3.add_argument("--use_logreg", action="store_true", help="Usa LR regularizada em vez de média ponderada")

    spI = sub.add_parser("infer", help="Roda inferência fim-a-fim")
    spI.add_argument("--domains", nargs="+", required=True, help="Lista de diretórios de domínio usados no Step 2")
    spI.add_argument("--final_dir", required=True, help="Diretório do Step 3 com final_vote_meta.pkl")
    spI.add_argument("--input_csv", required=True, help="CSV de entrada com colunas KOI/TOI/K2 (o que existir)")
    spI.add_argument("--out_csv", required=True, help="CSV de saída com probabilidades e predições")

    args = ap.parse_args()
    if args.cmd == "step2":
        build_local_voter(Path(args.run_dir), args.domain)
    elif args.cmd == "step3":
        ddirs = [Path(d) for d in args.domains]
        build_step3_multi(ddirs, Path(args.out_dir), mode=args.mode, alpha=args.alpha,
                          cap_ratio=args.cap_ratio, small_n=args.small_n,
                          temp_small=args.temp_small, use_logreg=args.use_logreg, rnd=RND)
    elif args.cmd == "infer":
        ddirs = [Path(d) for d in args.domains]
        infer(ddirs, Path(args.final_dir), Path(args.input_csv), Path(args.out_csv))

if __name__ == "__main__":
    main()
