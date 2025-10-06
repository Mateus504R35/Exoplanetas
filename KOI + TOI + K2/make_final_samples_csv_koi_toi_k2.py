#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera um CSV único alinhado para treinar o meta-modelo, agora com KOI, TOI e K2.

Alinhamento de colunas (fixo):
 KOI: koi_disposition, koi_period, koi_duration, koi_depth, koi_prad, koi_insol, koi_teq
 TOI: tfopwg_disp,     pl_orbper,  pl_trandurh,  pl_trandep, pl_rade,  pl_insol, pl_eqt, st_teff, st_logg
 K2 : <disp_col>,      pl_orbper,  pl_trandur,   pl_trandep, pl_rade,  pl_insol, pl_eqt, st_teff, st_logg

- Lê CSVs em: koi_run/, toi_run/ e k2_run/ (por padrão; pode trocar via args)
- Salva: final_samples_for_meta.csv (ou nome passado em --out_name)

Rótulo binário:
  planeta = 1  (KOI: CONFIRMED; TOI: CP/KP; K2: CONFIRMED)
  n_planeta = 0 (KOI: FALSE POSITIVE; TOI: FP/FA; K2: FALSE POSITIVE)
  candidato/ambíguo -> NaN (KOI: CANDIDATE; TOI: PC/APC; K2: CANDIDATE/REFUTED etc.)
"""
import argparse
from pathlib import Path
import pandas as pd

# ---- colunas alvo por domínio ----
K_TARGET_COLS = ["koi_period","koi_duration","koi_depth","koi_prad","koi_insol","koi_teq"]
T_TARGET_COLS = ["pl_orbper","pl_trandurh","pl_trandep","pl_rade","pl_insol","pl_eqt","st_teff","st_logg"]
K2_TARGET_COLS = ["pl_orbper","pl_trandur","pl_trandep","pl_rade","pl_insol","pl_eqt","st_teff","st_logg"]

# ---- mapeamentos diretos ----
KOI_MAP = {
    "koi_period":  "koi_period",
    "koi_duration":"koi_duration",
    "koi_depth":   "koi_depth",
    "koi_prad":    "koi_prad",
    "koi_insol":   "koi_insol",
    "koi_teq":     "koi_teq",
    "__label__":   "koi_disposition",
}
TOI_MAP = {
    "pl_orbper":   "pl_orbper",
    "pl_trandurh": "pl_trandurh",
    "pl_trandep":  "pl_trandep",
    "pl_rade":     "pl_rade",
    "pl_insol":    "pl_insol",
    "pl_eqt":      "pl_eqt",
    "st_teff":     "st_teff",
    "st_logg":     "st_logg",
    "__label__":   "tfopwg_disp",
}

# K2 pode variar nomes; este é o alvo padrão. A coluna de rótulo será autodetectada.
K2_MAP_DEFAULT = {
    "pl_orbper":   "pl_orbper",
    "pl_trandur":  "pl_trandur",
    "pl_trandep":  "pl_trandep",
    "pl_rade":     "pl_rade",
    "pl_insol":    "pl_insol",
    "pl_eqt":      "pl_eqt",
    "st_teff":     "st_teff",
    "st_logg":     "st_logg",
    "__label__":   None,  # autodetect
}

K2_LABEL_CANDIDATES = [
    "k2_disposition","disposition","k2_disp","tfopwg_disp","koi_disposition"
]

def script_dir() -> Path:
    try: return Path(__file__).resolve().parent
    except NameError: return Path.cwd()

def find_csv_in_dir(dir_path: Path, prefer_keyword: str):
    csvs = sorted(dir_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nenhum .csv encontrado em: {dir_path}")
    prefer = [p for p in csvs if prefer_keyword.lower() in p.name.lower()]
    if len(prefer) == 1:
        return prefer[0]
    if len(prefer) > 1:
        return max(prefer, key=lambda p: p.stat().st_size)
    if len(csvs) == 1:
        return csvs[0]
    return max(csvs, key=lambda p: p.stat().st_size)

def robust_read_csv(path: Path) -> pd.DataFrame:
    errors = []
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, engine="python", sep=None, comment="#",
                             skip_blank_lines=True, on_bad_lines="skip", encoding=enc)
            if df.shape[1] == 1:
                df = pd.read_csv(path, engine="python", sep=";", comment="#",
                                 skip_blank_lines=True, on_bad_lines="skip", encoding=enc)
            return df
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise RuntimeError(f"Falha ao ler {path}. Tentativas: {' | '.join(errors)}")

def detect_k2_label_col(df: pd.DataFrame):
    cols_upper = {c.upper(): c for c in df.columns}
    for cand in K2_LABEL_CANDIDATES:
        if cand.upper() in cols_upper:
            return cols_upper[cand.upper()]
    hits = [c for c in df.columns if "disp" in c.lower() or "disposition" in c.lower()]
    return hits[0] if hits else None

def normalize_block(df, mapping: dict, target_cols: list[str], domain_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for tgt in target_cols:
        src = mapping.get(tgt, None)
        out[tgt] = df[src] if (src is not None and src in df.columns) else pd.NA

    # rótulo (binário: 1 planeta, 0 falso positivo, NaN candidato/ambíguo)
    lab_col = mapping.get("__label__", None)
    if lab_col is None:
        out["label"] = pd.NA
    else:
        if lab_col in df.columns:
            lab = df[lab_col].astype(str).str.upper().str.strip()
            lab = lab.replace({
                # KOI
                "CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": pd.NA,
                # TOI
                "CP": 1, "KP": 1, "FP": 0, "FA": 0, "PC": pd.NA, "APC": pd.NA,
                # K2 (comuns)
                "CONFIRMADO": 1, "FALSO POSITIVO": 0, "REFUTED": pd.NA
            })
            out["label"] = pd.to_numeric(lab, errors="coerce")
        else:
            out["label"] = pd.NA

    out["source"] = domain_name
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koi_dir", default="koi_run", help="Pasta com o CSV do KOI")
    ap.add_argument("--toi_dir", default="toi_run", help="Pasta com o CSV do TOI")
    ap.add_argument("--k2_dir",  default="k2_run",  help="Pasta com o CSV do K2")
    ap.add_argument("--out_name", default="final_samples_for_meta.csv", help="Nome do CSV de saída")
    args = ap.parse_args()

    base_koi = Path(args.koi_dir).resolve()
    base_toi = Path(args.toi_dir).resolve()
    base_k2  = Path(args.k2_dir).resolve()
    out_path = script_dir() / args.out_name

    koi_csv = find_csv_in_dir(base_koi, prefer_keyword="koi")
    toi_csv = find_csv_in_dir(base_toi, prefer_keyword="toi")
    k2_csv  = find_csv_in_dir(base_k2,  prefer_keyword="k2")
    print(f"[INFO] KOI CSV: {koi_csv}")
    print(f"[INFO] TOI CSV: {toi_csv}")
    print(f"[INFO] K2  CSV: {k2_csv}")
    print(f"[INFO] Saída: {out_path}")

    koi_raw = robust_read_csv(koi_csv)
    toi_raw = robust_read_csv(toi_csv)
    k2_raw  = robust_read_csv(k2_csv)

    # --- KOI ---
    koi_norm = normalize_block(koi_raw, KOI_MAP, K_TARGET_COLS, "KOI")

    # --- TOI ---
    toi_norm = normalize_block(toi_raw, TOI_MAP, T_TARGET_COLS, "TOI")

    # --- K2 (autodetecta coluna de rótulo) ---
    k2_map = K2_MAP_DEFAULT.copy()
    k2_map["__label__"] = detect_k2_label_col(k2_raw)
    if k2_map["__label__"] is None:
        print("[WARN] Não encontrei coluna de disposição no K2; 'label' ficará NaN.")
    k2_norm = normalize_block(k2_raw, k2_map, K2_TARGET_COLS, "K2")

    # garantir colunas finais
    all_cols = K_TARGET_COLS + T_TARGET_COLS + K2_TARGET_COLS + ["label","source"]
    for df_norm in (koi_norm, toi_norm, k2_norm):
        for c in all_cols:
            if c not in df_norm.columns:
                df_norm[c] = pd.NA

    final_df = pd.concat([koi_norm[all_cols], toi_norm[all_cols], k2_norm[all_cols]], ignore_index=True)

    # remove linhas sem nenhuma feature preenchida
    feat_cols = K_TARGET_COLS + T_TARGET_COLS + K2_TARGET_COLS
    final_df = final_df[final_df[feat_cols].notna().any(axis=1)].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    print(f"[OK] CSV gerado: {out_path}")
    print(f"[OK] Shape: {final_df.shape}")
    print(final_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
