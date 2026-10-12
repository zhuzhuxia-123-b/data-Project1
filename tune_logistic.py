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


MAX_MODEL_ITERATIONS = 2000

# A、B 用于检查词表大小；
# B、C、D E 用于比较 C。
EXPERIMENTS = [
    {
        "experiment": "A_vocab5000_C1",
        "max_features": 5000,
        "C": 1.0,
    },
    {
        "experiment": "B_vocab20000_C1",
        "max_features": 20000,
        "C": 1.0,
    },
    {
        "experiment": "C_vocab20000_C0.1",
        "max_features": 20000,
        "C": 0.1,
    },
    {
        "experiment": "D_vocab20000_C10",
        "max_features": 20000,
        "C": 10.0,
    },
    {
        "experiment": "E_vocab20000_C100",
        "max_features": 20000,
        "C": 100.0,
    },
]


def get_actual_n_iter(model):
    """返回 Logistic Regression 实际使用的最大迭代次数。"""
    n_iter_array = np.asarray(model.n_iter_)
    return int(n_iter_array.max())


def main():
    print("=== Logistic Regression 参数实验 ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"验证集比例: {VALIDATION_SIZE}")
    print("固定 ngram_range=(1, 1)")
    print("固定 sublinear_tf=False")
    print("固定 solver='lbfgs'")
    print("固定 L2 正则化")
    print(f"模型最大迭代次数: {MAX_MODEL_ITERATIONS}")

    X, y = load_training_data()

    # 与 SVM 实验使用完全相同的数据划分。
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

    # 同一种 TF-IDF 配置只拟合一次。
    # key 是 max_features，value 保存训练和验证特征。
    feature_cache = {}
    feature_times = {}

    for max_features in sorted({
        config["max_features"]
        for config in EXPERIMENTS
    }):
        print(
            f"\n正在构建 max_features={max_features} "
            "的 TF-IDF 特征……"
        )

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            sublinear_tf=False,
            lowercase=True,
            stop_words=None,
        )

        feature_start = perf_counter()

        # 只在训练文本上拟合。
        X_train_tfidf = vectorizer.fit_transform(
            X_train
        )

        # 验证文本只能 transform。
        X_val_tfidf = vectorizer.transform(
            X_val
        )

        feature_seconds = (
            perf_counter() - feature_start
        )

        feature_cache[max_features] = (
            X_train_tfidf,
            X_val_tfidf,
        )
        feature_times[max_features] = (
            feature_seconds
        )

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

    records = []

    for config in EXPERIMENTS:
        experiment = config["experiment"]
        max_features = config["max_features"]
        c_value = config["C"]

        print(f"\n=== {experiment} ===")
        print(f"max_features={max_features}")
        print(f"C={c_value}")

        (
            X_train_tfidf,
            X_val_tfidf,
        ) = feature_cache[max_features]

        # l1_ratio=0.0 表示使用 L2 正则化。
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            l1_ratio=0.0,
            max_iter=MAX_MODEL_ITERATIONS,
            random_state=RANDOM_STATE,
        )

        fit_start = perf_counter()

        # 本轮记录收敛情况。
        # 如果未收敛，继续保存结果，但不能直接将其
        # 当作可靠的最终参数比较结果。
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

        fit_seconds = (
            perf_counter() - fit_start
        )

        convergence_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(
                warning.category,
                ConvergenceWarning,
            )
        ]

        converged = (
            len(convergence_warnings) == 0
        )

        warning_message = " | ".join(
            str(warning.message)
            for warning in convergence_warnings
        )

        actual_n_iter = get_actual_n_iter(
            model
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

        record = {
            "experiment": experiment,
            "max_features": max_features,
            "ngram_range": "(1, 1)",
            "sublinear_tf": False,
            "C": c_value,
            "solver": "lbfgs",
            "regularization": "L2",
            "max_iter": MAX_MODEL_ITERATIONS,
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
            "feature_seconds": (
                feature_times[max_features]
            ),
            "fit_seconds": fit_seconds,
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "sklearn_version": (
                sklearn.__version__
            ),
            "model_params": str(
                model.get_params()
            ),
        }
        records.append(record)

        status = (
            "已收敛"
            if converged
            else "未收敛"
        )

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
            f"{accuracy_gap * 100:.2f} "
            "个百分点"
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

    results = pd.DataFrame(records)

    print(
        "\n=== 特征对照："
        "C=1 时 5000 与 20000 维比较 ==="
    )

    feature_comparison = results.loc[
        results["C"] == 1.0
    ].sort_values(
        by="max_features"
    )

    feature_columns = [
        "experiment",
        "max_features",
        "actual_n_iter",
        "converged",
        "train_accuracy",
        "val_accuracy",
        "val_macro_f1",
        "accuracy_gap",
    ]

    print(
        feature_comparison[
            feature_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print(
        "\n=== C 参数对照："
        "固定 max_features=20000 ==="
    )

    c_comparison = results.loc[
        results["max_features"] == 20000
    ].sort_values(
        by=[
            "val_macro_f1",
            "val_accuracy",
            "C",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    c_columns = [
        "C",
        "actual_n_iter",
        "converged",
        "train_accuracy",
        "train_macro_f1",
        "val_accuracy",
        "val_macro_f1",
        "accuracy_gap",
        "fit_seconds",
    ]

    print(
        c_comparison[
            c_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    # 仅在已收敛的实验中选择候选配置。
    converged_results = results.loc[
        results["converged"]
    ].copy()

    if converged_results.empty:
        print(
            "\n没有任何配置完成收敛，"
            "本轮不能选择最佳候选配置。"
        )
    else:
        ranked = converged_results.sort_values(
            by=[
                "val_macro_f1",
                "val_accuracy",
                "max_features",
                "C",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        ).reset_index(drop=True)

        best = ranked.iloc[0]

        print(
            "\n=== 已尝试且已收敛配置中的"
            "最佳 LR 候选 ==="
        )
        print(
            f"实验: {best['experiment']}"
        )
        print(
            "max_features: "
            f"{int(best['max_features'])}"
        )
        print(f"C: {best['C']:g}")
        print(
            "Validation Accuracy: "
            f"{best['val_accuracy']:.4f}"
        )
        print(
            "Validation Macro-F1: "
            f"{best['val_macro_f1']:.4f}"
        )

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / "logistic_c_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 已保存 ===")
    print(f"参数实验结果: {output_path}")
    print(
        "重复运行本脚本会更新该文件。"
    )


if __name__ == "__main__":
    main()
    