from time import perf_counter
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from baseline import (
    DATA_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)
from compare_models import calculate_metrics


ITERATION_CHECKPOINTS = [
    1,
    2,
    3,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
]

LOGISTIC_C = 100.0


def get_actual_n_iter(model):
    """返回求解器实际使用的最大迭代次数。"""
    n_iter_array = np.asarray(model.n_iter_)
    return int(n_iter_array.max())


def main():
    print(
        "=== Logistic Regression "
        "log-loss checkpoint 实验 ==="
    )
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"验证集比例: {VALIDATION_SIZE}")
    print(
        f"迭代预算检查点: "
        f"{ITERATION_CHECKPOINTS}"
    )

    print("\n固定 TF-IDF:")
    print("max_features=20000")
    print("ngram_range=(1, 1)")
    print("sublinear_tf=False")

    print("\n固定 Logistic Regression:")
    print(f"C={LOGISTIC_C}")
    print("solver='lbfgs'")
    print("regularization=L2")

    X, y = load_training_data()

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
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

    # 只在训练集上拟合 TF-IDF。
    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )
    X_val_tfidf = vectorizer.transform(
        X_val
    )

    feature_seconds = (
        perf_counter() - feature_start
    )

    print("\n=== TF-IDF 特征 ===")
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

    for max_iter in ITERATION_CHECKPOINTS:
        print(f"\n=== max_iter={max_iter} ===")

        # 每个检查点都独立创建并训练模型。
        model = LogisticRegression(
            C=LOGISTIC_C,
            solver="lbfgs",
            l1_ratio=0.0,
            max_iter=max_iter,
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

        # predict_proba 返回每个类别的概率。
        train_probabilities = (
            model.predict_proba(
                X_train_tfidf
            )
        )
        val_probabilities = (
            model.predict_proba(
                X_val_tfidf
            )
        )

        # 计算多分类交叉熵。
        # 这里只包含数据拟合损失，
        # 不包含模型内部的 L2 正则化项。
        train_log_loss = log_loss(
            y_train,
            train_probabilities,
            labels=model.classes_,
        )
        val_log_loss = log_loss(
            y_val,
            val_probabilities,
            labels=model.classes_,
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

        record = {
            "max_iter": max_iter,
            "actual_n_iter": actual_n_iter,
            "converged": converged,
            "convergence_warning": (
                warning_message
            ),
            "train_log_loss": (
                train_log_loss
            ),
            "val_log_loss": val_log_loss,
            "train_accuracy": (
                train_metrics["accuracy"]
            ),
            "val_accuracy": (
                val_metrics["accuracy"]
            ),
            "train_macro_f1": (
                train_metrics["macro_f1"]
            ),
            "val_macro_f1": (
                val_metrics["macro_f1"]
            ),
            "accuracy_gap": (
                train_metrics["accuracy"]
                - val_metrics["accuracy"]
            ),
            "fit_seconds": fit_seconds,
            "C": LOGISTIC_C,
            "solver": "lbfgs",
            "regularization": "L2",
            "max_features": 20000,
            "ngram_range": "(1, 1)",
            "sublinear_tf": False,
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "sklearn_version": (
                sklearn.__version__
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
            f"Train log loss: "
            f"{train_log_loss:.6f}"
        )
        print(
            f"Validation log loss: "
            f"{val_log_loss:.6f}"
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
            f"分类器训练耗时: "
            f"{fit_seconds:.3f} 秒"
        )

        if warning_message:
            print(
                "捕获的 ConvergenceWarning: "
                f"{warning_message}"
            )

    results = pd.DataFrame(records)
    results = results.sort_values(
        by="max_iter"
    ).reset_index(drop=True)

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    csv_path = (
        output_dir
        / "logistic_loss_history.csv"
    )

    results.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 直接绘制真实检查点，不进行平滑。
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        results["max_iter"],
        results["train_log_loss"],
        marker="o",
        linewidth=1.8,
        label="Train log loss",
    )

    ax.plot(
        results["max_iter"],
        results["val_log_loss"],
        marker="s",
        linewidth=1.8,
        label="Validation log loss",
    )

    ax.set_xscale("log")
    ax.set_xticks(ITERATION_CHECKPOINTS)
    ax.get_xaxis().set_major_formatter(
        ScalarFormatter()
    )

    ax.set_title(
        "Logistic Regression Loss at "
        "Iteration-Budget Checkpoints"
    )
    ax.set_xlabel(
        "Maximum iteration budget "
        "(max_iter, log scale)"
    )
    ax.set_ylabel(
        "Multiclass log loss"
    )
    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.35,
    )
    ax.legend()

    fig.tight_layout()

    figure_path = (
        output_dir
        / "logistic_loss_curve.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n=== 汇总结果 ===")

    display_columns = [
        "max_iter",
        "actual_n_iter",
        "converged",
        "train_log_loss",
        "val_log_loss",
        "train_accuracy",
        "val_accuracy",
        "train_macro_f1",
        "val_macro_f1",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )

    print("\n=== 已保存 ===")
    print(f"检查点数据: {csv_path}")
    print(f"损失曲线: {figure_path}")

    print(
        "\n注意：图中是不同 max_iter "
        "预算下独立重新训练得到的 "
        "log-loss checkpoint curve，"
        "不是单次 fit 自动输出的"
        "原生逐迭代历史。"
    )
    print(
        "图中 log loss 不包含 L2 "
        "正则化项，因此不是完整优化目标。"
    )


if __name__ == "__main__":
    main()