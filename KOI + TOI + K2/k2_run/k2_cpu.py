#!/usr/bin/env python3
# ============================================================
# K2 - Treinamento CPU a partir de dados pré-processados
# *** Versão com tratamento de desbalanceamento (igual TOI) ***
# - RF e LR com class_weight="balanced"
# - XGB com scale_pos_weight = (#neg / #pos) e otimização por CV
# Espera preprocessed/data_ready.npz com X_train/X_test/y_train/y_test/feature_names
# e feature_names contidas em:
#   ['pl_orbper','pl_trandur','pl_trandep','pl_rade','pl_insol','pl_eqt','st_teff','st_logg']
# ============================================================
import os
import argparse
import numpy as np
import joblib
import json

from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

K2_FULL_FEATURES = [
    "pl_orbper",
    "pl_trandur",
    "pl_trandep",
    "pl_rade",
    "pl_insol",
    "pl_eqt",
    "st_teff",
    "st_logg",
]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_npz(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feat_names = data["feature_names"].tolist()
    return X_train, X_test, y_train, y_test, feat_names

def assert_k2_features(feat_names, strict: bool = False):
    """
    strict=False (padrão): aceita SUBCONJUNTO das K2_FULL_FEATURES (pós-VarianceThreshold).
    strict=True: exige que as features correspondam exatamente às K2_FULL_FEATURES naquela ordem.
    """
    feats = list(map(str, feat_names))
    if strict:
        if len(feats) != len(K2_FULL_FEATURES) or any(a != b for a, b in zip(feats, K2_FULL_FEATURES)):
            raise ValueError(
                "As features do NPZ não batem com o conjunto fixo esperado de K2.\n"
                f"Esperado (ordem exata): {K2_FULL_FEATURES}\n"
                f"Encontrado: {feats}"
            )
    else:
        unknown = [c for c in feats if c not in K2_FULL_FEATURES]
        if unknown:
            raise ValueError(
                "Foram encontradas features desconhecidas no NPZ em relação ao conjunto K2 esperado.\n"
                f"K2 esperado (superset): {K2_FULL_FEATURES}\n"
                f"Encontrado: {feats}\n"
                f"Desconhecidas: {unknown}"
            )
    return feats

def compute_scale_pos_weight(y):
    y = np.asarray(y).astype(int)
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0:
        return 1.0
    return max(neg / float(pos), 1.0)

def hyperparam_spaces_rf_xgb(base_spw: float):
    rf_param_dist = {
        "n_estimators": randint(300, 1601),
        "max_depth": randint(5, 31),
        "min_samples_split": randint(2, 21),
        "min_samples_leaf": randint(1, 11),
        "max_features": ["sqrt", "log2", None]
    }
    low = max(base_spw * 0.5, 1e-6)
    width = base_spw * 1.0  # uniform(low, width) -> [0.5*spw, 1.5*spw]
    xgb_param_dist = {
        "n_estimators": randint(600, 2001),
        "learning_rate": uniform(0.01, 0.29),
        "max_depth": randint(3, 11),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 11),
        "reg_lambda": uniform(1.0, 19.0),
        "scale_pos_weight": uniform(low, width),
    }
    return rf_param_dist, xgb_param_dist

def eval_model(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print(f"\n=== {name} ===")
    print("Matriz de confusão:")
    print(cm)
    print(f"Acurácia:    {acc:.4f}")
    print(f"Sensibilidade (Recall/TPR): {rec:.4f}")
    print(f"Especificidade (TNR):       {spec:.4f}")
    print(f"Precisão:    {prec:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_te, y_pred, digits=4))

def main():
    parser = argparse.ArgumentParser(description="Treino K2 CPU (com tratamento de desbalanceamento) a partir de dados pré-processados")
    parser.add_argument("--data_npz", type=str, default=os.path.join("preprocessed", "data_ready.npz"),
                        help="Caminho para o NPZ gerado pelo k2_preprocess.py.")
    parser.add_argument("--model_dir", type=str, default="models_cpu", help="Diretório para salvar .pkl dos modelos.")
    parser.add_argument("--n_iter_rf", type=int, default=25, help="Iterações de busca aleatória p/ RF.")
    parser.add_argument("--n_iter_xgb", type=int, default=35, help="Iterações de busca aleatória p/ XGB.")
    parser.add_argument("--strict_features", action="store_true",
                        help="Se ligado, exige ordem e igualdade exata das features (antes de VT).")
    args = parser.parse_args()

    ensure_dir(args.model_dir)

    # 1) Carregar dados prontos
    X_train, X_test, y_train, y_test, feat_names = load_npz(args.data_npz)
    feats_checked = assert_k2_features(feat_names, strict=args.strict_features)

    print("Shapes:", X_train.shape, X_test.shape)
    print("Nº de features:", len(feats_checked))
    print("Feature names:", feats_checked)

    # 2) Calcular desbalanceamento na base de treino
    spw = compute_scale_pos_weight(y_train)
    print(f"\n>> scale_pos_weight (base): {spw:.4f}  [= #neg / #pos no conjunto de treino]")

    # 3) Modelos base com tratamento de desbalanceamento
    rf_clf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        nthread=-1,
        scale_pos_weight=spw
    )

    # 4) Espaços de hiperparâmetros
    rf_param_dist, xgb_param_dist = hyperparam_spaces_rf_xgb(spw)

    # 5) Validação cruzada e busca
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    f1_scorer = make_scorer(f1_score)

    print("\n>>> Ajustando Random Forest (CV 5x5) [class_weight=balanced] ...")
    rf_search = RandomizedSearchCV(
        rf_clf, rf_param_dist, n_iter=args.n_iter_rf, scoring=f1_scorer,
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=2
    )
    rf_search.fit(X_train, y_train)
    print("Melhores params RF:", rf_search.best_params_)
    print("Melhor F1 (CV) RF:", rf_search.best_score_)

    print("\n>>> Ajustando XGBoost (CV 5x5) [scale_pos_weight otimizado] ...")
    xgb_search = RandomizedSearchCV(
        xgb_clf, xgb_param_dist, n_iter=args.n_iter_xgb, scoring=f1_scorer,
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=2
    )
    xgb_search.fit(X_train, y_train)
    print("Melhores params XGB:", xgb_search.best_params_)
    print("Melhor F1 (CV) XGB:", xgb_search.best_score_)

    rf_best = rf_search.best_estimator_
    xgb_best = xgb_search.best_estimator_

    # 6) Stacking (com LR balanceada)
    stack_clf = StackingClassifier(
        estimators=[("rf", rf_best), ("xgb", xgb_best)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="auto",
        n_jobs=-1,
        passthrough=False,
        cv=5
    )
    print("\n>>> Ajustando Stacking (RF + XGB -> LR[class_weight=balanced]) ...")
    stack_clf.fit(X_train, y_train)

    # 7) Avaliar
    eval_model("Random Forest (melhor CV)", rf_best, X_test, y_test)
    eval_model("XGBoost (melhor CV)", xgb_best, X_test, y_test)
    eval_model("Stacking (RF + XGB -> LR)", stack_clf, X_test, y_test)

    # 8) Salvar modelos
    paths = {
        "rf_model": os.path.join(args.model_dir, "rf_model.pkl"),
        "xgb_model": os.path.join(args.model_dir, "xgb_model.pkl"),
        "stack_model": os.path.join(args.model_dir, "stack_model.pkl"),
    }
    joblib.dump(rf_best, paths["rf_model"])
    joblib.dump(xgb_best, paths["xgb_model"])
    joblib.dump(stack_clf, paths["stack_model"])

    meta = {
        "feature_names": feats_checked,
        "n_features": len(feats_checked),
        "random_state": RANDOM_STATE,
        "cv": "RepeatedStratifiedKFold(5x5)",
        "search_iters": {"rf": int(args.n_iter_rf), "xgb": int(args.n_iter_xgb)},
        "class_balance": {
            "scale_pos_weight_base": float(spw),
            "rf_class_weight": "balanced",
            "stack_lr_class_weight": "balanced"
        }
    }
    with open(os.path.join(args.model_dir, "cpu_results_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nModelos salvos:")
    for k, v in paths.items():
        print(f" - {k}: {v}")
    print(f"Metadados: {os.path.join(args.model_dir, 'cpu_results_meta.json')}")

if __name__ == "__main__":
    main()
