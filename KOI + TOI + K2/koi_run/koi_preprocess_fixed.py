#!/usr/bin/env python3
# ============================================================
# KOI (NASA) - ETL/Pré-processamento -> dados prontos p/ treino
# Agora com UM CONJUNTO FIXO de features numéricas:
#   - koi_period   (Período orbital, dias)
#   - koi_duration (Duração do trânsito)
#   - koi_depth    (Profundidade do trânsito)
#   - koi_prad     (Raio planetário, R⊕)
#   - koi_insol    (Insolação)
#   - koi_teq      (Temperatura de equilíbrio)
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

# Conjunto FIXO de colunas (KOI)
FEATURE_COLS = [
    "koi_period",   # Período orbital (dias)
    "koi_duration", # Duração do trânsito
    "koi_depth",    # Profundidade do trânsito
    "koi_prad",     # Raio planetário (R⊕)
    "koi_insol",    # Insolação
    "koi_teq"       # Temperatura de equilíbrio
]
TARGET_COL = "koi_disposition"  # CONFIRMED/ FALSE POSITIVE -> 1/0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv(csv_path: str) -> pd.DataFrame:
    """Lê CSV KOI ignorando metadados iniciados por '#'."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    df_raw = pd.read_csv(csv_path, comment="#")
    return df_raw


def clean_and_select(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Seleciona apenas FEATURE_COLS + TARGET, trata rótulo e valores inválidos."""
    import numpy as _np

    # Binariza alvo se existir
    if TARGET_COL in df_raw.columns:
        df = df_raw.copy()
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.upper()
        # Mantém apenas CONFIRMED e FALSE POSITIVE
        df = df[df[TARGET_COL].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
        # CONFIRMED = 1, FALSE POSITIVE = 0
        df[TARGET_COL] = _np.where(df[TARGET_COL] == "CONFIRMED", 1, 0).astype(int)
    else:
        raise ValueError(f"É necessário conter a coluna-alvo '{TARGET_COL}' no CSV para gerar y_train/y_test.")

    # Garante que TODAS as features existam
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"As seguintes colunas exigidas estão ausentes no CSV: {missing}")

    # Seleciona e força numérico
    use_cols = FEATURE_COLS + [TARGET_COL]
    df = df[use_cols].copy()
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Trata infs -> NaN
    df.replace([_np.inf, -_np.inf], _np.nan, inplace=True)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Pipeline numérico para as 6 features fixas."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURE_COLS)],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


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
    pd.Series(y_train, name=TARGET_COL).to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    pd.Series(y_test, name=TARGET_COL).to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
    Xtr_df.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    Xte_df.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Pré-processamento KOI -> dados prontos para treino (features fixas)")
    parser.add_argument("--csv", required=True, type=str, help="Caminho para o CSV bruto (KOI).")
    parser.add_argument("--out_dir", default="preprocessed", type=str, help="Diretório de saída.")
    parser.add_argument("--test_size", default=0.30, type=float, help="Proporção do conjunto de teste.")
    parser.add_argument("--random_state", default=RANDOM_STATE, type=int, help="Seed para reprodutibilidade.")
    parser.add_argument("--save_csv", action="store_true", help="Também salvar X_train/X_test/y_*.csv para inspeção.")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 1) Carregar e selecionar
    df_raw = load_csv(args.csv)
    df = clean_and_select(df_raw)

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # 3) Pré-processamento (fit apenas no treino) + VarianceThreshold (por segurança)
    pre = build_preprocessor()
    Xtr_pre = pre.fit_transform(X_train)
    Xte_pre = pre.transform(X_test)

    # Nomes após pre-process (iguais às FEATURE_COLS, pois não há OneHot)
    feat_after_pre = list(FEATURE_COLS)

    vt = VarianceThreshold(threshold=0.0)
    Xtr_final = vt.fit_transform(Xtr_pre)
    Xte_final = vt.transform(Xte_pre)

    # Nomes finais após VT
    support = vt.get_support()
    feat_final = [fname for fname, keep in zip(feat_after_pre, support) if keep]

    # 4) Salvar artefatos
    save_npz(args.out_dir, Xtr_final, Xte_final, y_train, y_test, feat_final)
    joblib.dump(pre, os.path.join(args.out_dir, "preprocessor.pkl"))
    joblib.dump(vt, os.path.join(args.out_dir, "vt.pkl"))

    # Metadados
    meta = {
        "random_state": args.random_state,
        "test_size": args.test_size,
        "X_train_shape": list(Xtr_final.shape),
        "X_test_shape": list(Xte_final.shape),
        "n_features": int(Xtr_final.shape[1]),
        "feature_cols": FEATURE_COLS
    }
    with open(os.path.join(args.out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feat_final, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.save_csv:
        save_csvs(args.out_dir, Xtr_final, Xte_final, y_train, y_test, feat_final)

    print("=== PRONTO ===")
    print(f"- NPZ: {os.path.join(args.out_dir, 'data_ready.npz')}")
    print(f"- feature_names.json, meta.json, preprocessor.pkl, vt.pkl salvos em: {args.out_dir}")
    if args.save_csv:
        print("- CSVs: X_train.csv, X_test.csv, y_train.csv, y_test.csv")


if __name__ == "__main__":
    main()
