#!/usr/bin/env python3
# ============================================================
# K2 (Planets and Candidates) - ETL/Pré-processamento
# Conjunto FIXO de features numéricas:
#   - pl_orbper   (Período orbital, dias)
#   - pl_trandur  (Duração do trânsito; K2 usa 'pl_trandur')
#   - pl_trandep  (Profundidade do trânsito)
#   - pl_rade     (Raio planetário, R⊕)
#   - pl_insol    (Insolação)
#   - pl_eqt      (Temperatura de equilíbrio)
#   - st_teff     (Temperatura efetiva da estrela)
#   - st_logg     (Gravidade superficial da estrela, log g)
#
# Rótulo (TARGET): disposition -> CONFIRMED=1 ; FALSE POSITIVE=0
# (CANDIDATE/REFUTED são descartados)
#
# Saídas: preprocessed/data_ready.npz, feature_names.json, meta.json,
#         preprocessor.pkl e vt.pkl
# ============================================================
import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURE_COLS = [
    "pl_orbper",
    "pl_trandur",   # K2: 'pl_trandur'
    "pl_trandep",
    "pl_rade",
    "pl_insol",
    "pl_eqt",
    "st_teff",
    "st_logg",
]
TARGET_COL = "disposition"   # CONFIRMED=1 ; FALSE POSITIVE=0
POS_SET = {"CONFIRMED"}
NEG_SET = {"FALSE POSITIVE"}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    # A tabela K2 do NEA costuma vir com cabeçalho comentado com '#'
    return pd.read_csv(csv_path, comment="#")

def clean_and_select(df_raw: pd.DataFrame) -> pd.DataFrame:
    import numpy as _np
    if TARGET_COL not in df_raw.columns:
        raise ValueError(f"É necessário conter a coluna '{TARGET_COL}' no CSV K2.")

    disp = df_raw[TARGET_COL].astype(str).str.upper().str.strip()
    # Mantém apenas CONFIRMED/FP; descarta CANDIDATE/REFUTED/etc.
    mask = disp.isin(POS_SET | NEG_SET)
    df = df_raw.loc[mask].copy()
    y = disp.loc[mask].apply(lambda v: 1 if v in POS_SET else 0).astype(int)
    df[TARGET_COL] = y

    # Verifica features obrigatórias
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"As seguintes colunas exigidas estão ausentes no CSV: {missing}")

    # Seleciona apenas as 8 features + alvo
    use_cols = FEATURE_COLS + [TARGET_COL]
    df = df[use_cols].copy()

    # Converte para numérico e trata inf/NaN
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([_np.inf, -_np.inf], _np.nan, inplace=True)
    return df

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, FEATURE_COLS)],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre

def save_npz(out_dir: str, X_train, X_test, y_train, y_test, feat_names):
    np.savez_compressed(
        os.path.join(out_dir, "data_ready.npz"),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=np.array(feat_names, dtype=object)
    )

def save_csvs(out_dir: str, X_train, X_test, y_train, y_test, feat_names):
    Xtr_df = pd.DataFrame(X_train, columns=feat_names)
    Xte_df = pd.DataFrame(X_test, columns=feat_names)
    pd.Series(y_train, name="label").to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    pd.Series(y_test, name="label").to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
    Xtr_df.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    Xte_df.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Pré-processamento K2 (features fixas) -> dados prontos para treino")
    parser.add_argument("--csv", required=True, type=str, help="Caminho para o CSV bruto (K2).")
    parser.add_argument("--out_dir", default="preprocessed", type=str, help="Diretório de saída.")
    parser.add_argument("--test_size", default=0.30, type=float, help="Proporção do conjunto de teste.")
    parser.add_argument("--random_state", default=RANDOM_STATE, type=int, help="Seed para reprodutibilidade.")
    parser.add_argument("--save_csv", action="store_true", help="Também salvar X_train/X_test/y_*.csv para inspeção.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df_raw = load_csv(args.csv)
    df = clean_and_select(df_raw)

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pre = build_preprocessor()
    Xtr_pre = pre.fit_transform(X_train)
    Xte_pre = pre.transform(X_test)

    feat_after_pre = list(FEATURE_COLS)

    vt = VarianceThreshold(threshold=0.0)
    Xtr_final = vt.fit_transform(Xtr_pre)
    Xte_final = vt.transform(Xte_pre)

    support = vt.get_support()
    feat_final = [fname for fname, keep in zip(feat_after_pre, support) if keep]

    save_npz(args.out_dir, Xtr_final, Xte_final, y_train, y_test, feat_final)
    joblib.dump(pre, os.path.join(args.out_dir, "preprocessor.pkl"))
    joblib.dump(vt, os.path.join(args.out_dir, "vt.pkl"))

    meta = {
        "random_state": args.random_state,
        "test_size": args.test_size,
        "X_train_shape": list(Xtr_final.shape),
        "X_test_shape": list(Xte_final.shape),
        "n_features": int(Xtr_final.shape[1]),
        "feature_cols": FEATURE_COLS,
        "label_mapping": {"CONFIRMED": 1, "FALSE POSITIVE": 0}
    }
    with open(os.path.join(args.out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feat_final, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.save_csv:
        save_csvs(args.out_dir, Xtr_final, Xte_final, y_train, y_test, feat_final)

    print("=== PRONTO (K2 - features fixas) ===")
    print(f"- NPZ: {os.path.join(args.out_dir, 'data_ready.npz')}")
    print(f"- feature_names.json, meta.json, preprocessor.pkl, vt.pkl salvos em: {args.out_dir}")
    if args.save_csv:
        print("- CSVs: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

if __name__ == "__main__":
    main()
