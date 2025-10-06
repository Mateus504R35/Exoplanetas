
#!/usr/bin/env python3
# ============================================================
# KOI (NASA) - Treinamento CPU a partir de dados pré-processados
# *** Versão FIXA (6 features) ***
# Espera preprocessed/data_ready.npz com X_train/X_test/y_train/y_test/feature_names
# e esses feature_names devem corresponder às 6 colunas:
#   - koi_period, koi_duration, koi_depth, koi_prad, koi_insol, koi_teq
# Não executa pré-processamento (apenas treinamento e avaliação)
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

FIXED_FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_insol",
    "koi_teq",
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


def assert_fixed_features(feat_names, strict: bool = True):
    # feat_names pode vir como list de str ou numpy array
    feats = list(map(str, feat_names))
    if strict:
        if len(feats) != len(FIXED_FEATURES) or any(a != b for a, b in zip(feats, FIXED_FEATURES)):
            raise ValueError(
                "As features do NPZ não batem com o conjunto fixo de 6 colunas.\n"
                f"Esperado (ordem exata): {FIXED_FEATURES}\n"
                f"Encontrado: {feats}"
            )
    else:
        missing = [c for c in FIXED_FEATURES if c not in feats]
        if missing:
            raise ValueError(f"NPZ não contém todas as features exigidas: ausentes={missing}")
    return feats


def hyperparam_spaces_rf_xgb():
    rf_param_dist = {
        "n_estimators": randint(300, 1601),
        "max_depth": randint(5, 31),
        "min_samples_split": randint(2, 21),
        "min_samples_leaf": randint(1, 11),
        "max_features": ["sqrt", "log2", None]
    }
    xgb_param_dist = {
        "n_estimators": randint(600, 2001),
        "learning_rate": uniform(0.01, 0.29),
        "max_depth": randint(3, 11),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 11),
        "reg_lambda": uniform(1.0, 19.0)
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
    parser = argparse.ArgumentParser(description="Treino KOI CPU (conjunto de 6 features fixas) a partir de dados pré-processados")
    parser.add_argument("--data_npz", type=str, default=os.path.join("preprocessed", "data_ready.npz"),
                        help="Caminho para o NPZ gerado pelo script de pré-processamento FIXO.")
    parser.add_argument("--model_dir", type=str, default="models_cpu", help="Diretório para salvar .pkl dos modelos.")
    parser.add_argument("--n_iter_rf", type=int, default=25, help="Iterações de busca aleatória p/ RF.")
    parser.add_argument("--n_iter_xgb", type=int, default=35, help="Iterações de busca aleatória p/ XGB.")
    parser.add_argument("--strict_features", action="store_true",
                        help="Se ligado, exige ordem e igualdade exata dos 6 nomes de features.")
    args = parser.parse_args()

    ensure_dir(args.model_dir)

    # 1) Carregar dados prontos
    X_train, X_test, y_train, y_test, feat_names = load_npz(args.data_npz)
    feats_checked = assert_fixed_features(feat_names, strict=args.strict_features)

    print("Shapes:", X_train.shape, X_test.shape)
    print("Features (esperado=6):", len(feats_checked))
    print("Feature names:", feats_checked)

    # 2) Modelos base
    rf_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        nthread=-1
    )

    # 3) Hiperparâmetros
    rf_param_dist, xgb_param_dist = hyperparam_spaces_rf_xgb()

    # 4) Validação cruzada e busca
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    f1_scorer = make_scorer(f1_score)

    print("\n>>> Ajustando Random Forest (CV 5x5)...")
    rf_search = RandomizedSearchCV(
        rf_clf, rf_param_dist, n_iter=args.n_iter_rf, scoring=f1_scorer,
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=2
    )
    rf_search.fit(X_train, y_train)
    print("Melhores params RF:", rf_search.best_params_)
    print("Melhor F1 (CV) RF:", rf_search.best_score_)

    print("\n>>> Ajustando XGBoost (CV 5x5)...")
    xgb_search = RandomizedSearchCV(
        xgb_clf, xgb_param_dist, n_iter=args.n_iter_xgb, scoring=f1_scorer,
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=2
    )
    xgb_search.fit(X_train, y_train)
    print("Melhores params XGB:", xgb_search.best_params_)
    print("Melhor F1 (CV) XGB:", xgb_search.best_score_)

    rf_best = rf_search.best_estimator_
    xgb_best = xgb_search.best_estimator_

    # 5) Stacking (sem pré-processamento aqui)
    stack_clf = StackingClassifier(
        estimators=[("rf", rf_best), ("xgb", xgb_best)],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="auto",
        n_jobs=-1,
        passthrough=False,
        cv=5
    )
    print("\n>>> Ajustando Stacking (RF + XGB -> LR)...")
    stack_clf.fit(X_train, y_train)

    # 6) Avaliar
    eval_model("Random Forest (melhor CV)", rf_best, X_test, y_test)
    eval_model("XGBoost (melhor CV)", xgb_best, X_test, y_test)
    eval_model("Stacking (RF + XGB -> LR)", stack_clf, X_test, y_test)

    # 7) Salvar modelos
    paths = {}
    paths["rf_model"] = os.path.join(args.model_dir, "rf_model.pkl")
    paths["xgb_model"] = os.path.join(args.model_dir, "xgb_model.pkl")
    paths["stack_model"] = os.path.join(args.model_dir, "stack_model.pkl")

    joblib.dump(rf_best, paths["rf_model"])
    joblib.dump(xgb_best, paths["xgb_model"])
    joblib.dump(stack_clf, paths["stack_model"])

    # Metadados úteis
    meta = {
        "feature_names": feats_checked,
        "n_features": len(feats_checked),
        "random_state": RANDOM_STATE,
        "cv": "RepeatedStratifiedKFold(5x5)",
        "search_iters": {"rf": int(args.n_iter_rf), "xgb": int(args.n_iter_xgb)}
    }
    with open(os.path.join(args.model_dir, "cpu_results_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nModelos salvos:")
    for k, v in paths.items():
        print(f" - {k}: {v}")
    print(f"Metadados: {os.path.join(args.model_dir, 'cpu_results_meta.json')}")


if __name__ == "__main__":
    main()
