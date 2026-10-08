from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from baseline import (
    DATA_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)
from compare_models import calculate_metrics


C_VALUE = 100.0
MAX_ITER = 2000


def main():
    print("=== Logistic Regression 补充参数实验 ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"C: {C_VALUE}")
    print("TF-IDF max_features: 20000")
    print("solver: lbfgs")
    print("regularization: L2")

    X, y = load_training_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n=== 数据划分 ===")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 1),
        sublinear_tf=False,
        lowercase=True,
        stop_words=None,
    )

    feature_start = perf_counter()

    # TF-IDF 只在训练集上拟合。
    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )
    X_val_tfidf = vectorizer.transform(
        X_val
    )

    feature_seconds = (
        perf_counter() - feature_start
    )

    print("\n=== TF-IDF ===")
    print(
        "训练特征矩阵形状: "
        f"{X_train_tfidf.shape}"
    )
    print(
        "验证特征矩阵形状: "
        f"{X_val_tfidf.shape}"
    )
    print(
        f"特征处理耗时: "
        f"{feature_seconds:.3f} 秒"
    )

    model = LogisticRegression(
        C=C_VALUE,
        solver="lbfgs",
        l1_ratio=0.0,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    fit_start = perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )
        model.fit(
            X_train_tfidf,
            y_train,
        )

    fit_seconds = perf_counter() - fit_start

    convergence_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]

    converged = len(convergence_warnings) == 0

    warning_message = " | ".join(
        str(warning.message)
        for warning in convergence_warnings
    )

    actual_n_iter = int(
        np.asarray(model.n_iter_).max()
    )

    train_predictions = model.predict(
        X_train_tfidf
    )
    val_predictions = model.predict(
        X_val_tfidf
    )

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )
    val_metrics = calculate_metrics(
        y_val,
        val_predictions,
    )

    accuracy_gap = (
        train_metrics["accuracy"]
        - val_metrics["accuracy"]
    )

    status = (
        "已收敛"
        if converged
        else "未收敛"
    )

    print("\n=== C=100 结果 ===")
    print(f"求解器状态: {status}")
    print(
        "实际迭代次数 n_iter_: "
        f"{actual_n_iter}"
    )
    print(
        "Train: "
        f"Accuracy="
        f"{train_metrics['accuracy']:.4f}, "
        f"Macro-F1="
        f"{train_metrics['macro_f1']:.4f}"
    )
    print(
        "Validation: "
        f"Accuracy="
        f"{val_metrics['accuracy']:.4f}, "
        f"Macro-F1="
        f"{val_metrics['macro_f1']:.4f}"
    )
    print(
        "Train-Val Accuracy gap: "
        f"{accuracy_gap * 100:.2f} 个百分点"
    )
    print(
        f"分类器训练耗时: "
        f"{fit_seconds:.3f} 秒"
    )

    if warning_message:
        print(
            "捕获的 ConvergenceWarning: "
            f"{warning_message}"
        )

    result = pd.DataFrame([{
        "experiment": "E_vocab20000_C100",
        "max_features": 20000,
        "ngram_range": "(1, 1)",
        "sublinear_tf": False,
        "C": C_VALUE,
        "solver": "lbfgs",
        "regularization": "L2",
        "max_iter": MAX_ITER,
        "actual_n_iter": actual_n_iter,
        "converged": converged,
        "convergence_warning": warning_message,
        "train_accuracy": (
            train_metrics["accuracy"]
        ),
        "train_macro_f1": (
            train_metrics["macro_f1"]
        ),
        "val_accuracy": (
            val_metrics["accuracy"]
        ),
        "val_macro_f1": (
            val_metrics["macro_f1"]
        ),
        "accuracy_gap": accuracy_gap,
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "sklearn_version": sklearn.__version__,
        "model_params": str(
            model.get_params()
        ),
    }])

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / "logistic_c100_result.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 已保存 ===")
    print(output_path)


if __name__ == "__main__":
    main()
    