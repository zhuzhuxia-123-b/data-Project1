from time import perf_counter
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from baseline import (
    DATA_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)
from compare_models import calculate_metrics


# 每个 max_iter 都会创建一个全新的模型并独立训练。
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

SVM_C = 1.0
SVM_TOL = 1e-4
LABELS = list(range(10))


def calculate_ovr_squared_hinge_loss(
    y_true,
    decision_scores,
    classes,
):
    """
    计算 one-vs-rest squared hinge 的数据损失。

    对每个样本和每个类别：
        正确类别的二分类标签为 +1；
        其他类别的二分类标签为 -1。

    每个样本在全部 one-vs-rest 分类器上的损失先求和，
    然后对全部样本取平均。

    该数值不包含 L2 正则化项，因此不是完整优化目标。
    """
    y_true_array = np.asarray(y_true)
    scores = np.asarray(decision_scores)
    classes_array = np.asarray(classes)

    if scores.ndim != 2:
        raise ValueError(
            "本实验应为 10 分类，decision_function 输出应为二维矩阵。"
        )

    expected_shape = (
        len(y_true_array),
        len(classes_array),
    )

    if scores.shape != expected_shape:
        raise ValueError(
            "decision_function 的形状与样本数或类别数不一致："
            f"实际为 {scores.shape}，期望为 {expected_shape}。"
        )

    # 生成 one-vs-rest 标签矩阵。
    # 真实类别对应位置为 +1，其余类别位置为 -1。
    signed_targets = np.where(
        y_true_array[:, np.newaxis]
        == classes_array[np.newaxis, :],
        1.0,
        -1.0,
    )

    margin_violations = np.maximum(
        0.0,
        1.0 - signed_targets * scores,
    )

    squared_losses = margin_violations ** 2

    # 每个样本先汇总 10 个 one-vs-rest 损失，
    # 再对所有样本取平均。
    loss = np.mean(np.sum(squared_losses, axis=1))

    return float(loss)


def get_actual_n_iter(model):
    """
    将 n_iter_ 安全转换为整数。

    当前 LinearSVC 通常返回一个整数；
    这里同时兼容返回数组的情况。
    """
    n_iter_array = np.asarray(model.n_iter_)
    return int(n_iter_array.max())


def main():
    print("=== LinearSVC loss checkpoint 实验 ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"验证集比例: {VALIDATION_SIZE}")
    print(f"迭代预算检查点: {ITERATION_CHECKPOINTS}")

    print("\n固定 TF-IDF:")
    print("max_features=20000")
    print("ngram_range=(1, 1)")
    print("sublinear_tf=False")

    print("\n固定 LinearSVC:")
    print(f"C={SVM_C}")
    print("penalty='l2'")
    print("loss='squared_hinge'")
    print("multi_class='ovr'")
    print("dual='auto'")
    print(f"tol={SVM_TOL}")

    X, y = load_training_data()

    # 复用之前完全相同的分层划分。
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

    # TF-IDF 只在训练文本上拟合一次。
    # 所有 checkpoint 使用完全相同的特征矩阵。
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 1),
        sublinear_tf=False,
        lowercase=True,
        stop_words=None,
    )

    feature_start = perf_counter()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    feature_seconds = perf_counter() - feature_start

    print("\n=== TF-IDF 特征 ===")
    print(f"训练特征矩阵形状: {X_train_tfidf.shape}")
    print(f"验证特征矩阵形状: {X_val_tfidf.shape}")
    print(f"特征处理耗时: {feature_seconds:.3f} 秒")

    records = []

    for max_iter in ITERATION_CHECKPOINTS:
        print(f"\n=== max_iter={max_iter} ===")

        # 每个 checkpoint 都创建并训练一个全新的模型。
        model = LinearSVC(
            C=SVM_C,
            penalty="l2",
            loss="squared_hinge",
            dual="auto",
            tol=SVM_TOL,
            max_iter=max_iter,
            multi_class="ovr",
            random_state=RANDOM_STATE,
        )

        fit_start = perf_counter()

        # ConvergenceWarning 在本实验中是需要记录的现象，
        # 因此捕获警告，但不终止程序。
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )
            model.fit(X_train_tfidf, y_train)

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

        actual_n_iter = get_actual_n_iter(model)

        # decision_function 给出每个 one-vs-rest 分类器的分数。
        train_scores = model.decision_function(
            X_train_tfidf
        )
        val_scores = model.decision_function(
            X_val_tfidf
        )

        train_loss = calculate_ovr_squared_hinge_loss(
            y_true=y_train,
            decision_scores=train_scores,
            classes=model.classes_,
        )
        val_loss = calculate_ovr_squared_hinge_loss(
            y_true=y_val,
            decision_scores=val_scores,
            classes=model.classes_,
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
            "convergence_warning": warning_message,
            "train_squared_hinge_loss": train_loss,
            "val_squared_hinge_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "fit_seconds": fit_seconds,
            "C": SVM_C,
            "tol": SVM_TOL,
            "loss": "squared_hinge",
            "penalty": "l2",
            "multi_class": "ovr",
            "dual": "auto",
            "max_features": 20000,
            "ngram_range": "(1, 1)",
            "sublinear_tf": False,
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "sklearn_version": sklearn.__version__,
        }
        records.append(record)

        status = "已收敛" if converged else "未收敛"

        print(f"求解器状态: {status}")
        print(f"实际迭代次数 n_iter_: {actual_n_iter}")
        print(
            "Train squared hinge loss: "
            f"{train_loss:.6f}"
        )
        print(
            "Validation squared hinge loss: "
            f"{val_loss:.6f}"
        )
        print(
            f"Train: Accuracy="
            f"{train_metrics['accuracy']:.4f}, "
            f"Macro-F1="
            f"{train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Validation: Accuracy="
            f"{val_metrics['accuracy']:.4f}, "
            f"Macro-F1="
            f"{val_metrics['macro_f1']:.4f}"
        )
        print(f"分类器训练耗时: {fit_seconds:.3f} 秒")

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

    csv_path = output_dir / "svm_loss_history.csv"
    results.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 使用原始 checkpoint 数据直接绘图，不进行平滑。
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        results["max_iter"],
        results["train_squared_hinge_loss"],
        marker="o",
        linewidth=1.8,
        label="Train squared hinge loss",
    )

    ax.plot(
        results["max_iter"],
        results["val_squared_hinge_loss"],
        marker="s",
        linewidth=1.8,
        label="Validation squared hinge loss",
    )

    # 检查点跨度较大，使用对数横轴方便观察。
    # 这只改变坐标显示，不改变任何实验数据。
    ax.set_xscale("log")
    ax.set_xticks(ITERATION_CHECKPOINTS)
    ax.get_xaxis().set_major_formatter(
        ScalarFormatter()
    )

    ax.set_title(
        "LinearSVC Loss at Iteration-Budget Checkpoints"
    )
    ax.set_xlabel(
        "Maximum iteration budget (max_iter, log scale)"
    )
    ax.set_ylabel(
        "Mean summed OvR squared hinge loss"
    )
    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.35,
    )
    ax.legend()

    fig.tight_layout()

    figure_path = output_dir / "svm_loss_curve.png"
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
        "train_squared_hinge_loss",
        "val_squared_hinge_loss",
        "train_accuracy",
        "val_accuracy",
        "train_macro_f1",
        "val_macro_f1",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n=== 已保存 ===")
    print(f"检查点数据: {csv_path}")
    print(f"损失曲线: {figure_path}")

    print(
        "\n注意：该图是不同 max_iter 预算下独立重新训练得到的"
        " checkpoint curve，不是单次 fit 内部的原生逐迭代历史。"
    )


if __name__ == "__main__":
    main()