from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from baseline import (
    DATA_DIR,
    VALIDATION_SIZE,
    load_training_data,
)
from compare_models import calculate_metrics


# 只改变数据划分的随机种子。
SEEDS = [42, 52, 62, 72, 82]

MAX_FEATURES = 20000
NGRAM_RANGE = (1, 1)

SVM_C = 1.0
LOGISTIC_C = 100.0


def get_actual_n_iter(model):
    """将模型的实际迭代次数转换成整数。"""
    n_iter_array = np.asarray(model.n_iter_)
    return int(n_iter_array.max())


def main():
    print("=== 多随机种子稳定性验证 ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {SEEDS}")
    print(f"验证集比例: {VALIDATION_SIZE}")

    print("\n固定TF-IDF配置：")
    print(f"max_features={MAX_FEATURES}")
    print(f"ngram_range={NGRAM_RANGE}")
    print("sublinear_tf=False")

    print("\n固定模型配置：")
    print(f"LinearSVC: C={SVM_C}")
    print(f"Logistic Regression: C={LOGISTIC_C}")
    print("本实验不读取测试集。")

    X, y = load_training_data()

    print("\n=== 数据 ===")
    print(f"有标签数据总数: {len(X)}")
    print(f"类别数: {y.nunique()}")

    records = []

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"随机种子: {seed}")

        # 每个随机种子都重新进行分层划分。
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            random_state=seed,
            stratify=y,
        )

        print(f"训练集样本数: {len(X_train)}")
        print(f"验证集样本数: {len(X_val)}")

        # 每次划分后重新拟合TF-IDF。
        # 验证集只执行transform，避免数据泄漏。
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=False,
            lowercase=True,
            stop_words=None,
        )

        feature_start = perf_counter()

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)

        feature_seconds = perf_counter() - feature_start

        print(
            "TF-IDF矩阵: "
            f"{X_train_tfidf.shape}, "
            f"特征处理耗时={feature_seconds:.3f}秒"
        )

        models = {
            "LinearSVC": LinearSVC(
                C=SVM_C,
                penalty="l2",
                loss="squared_hinge",
                dual="auto",
                max_iter=10000,
                random_state=seed,
            ),
            "LogisticRegression": LogisticRegression(
                C=LOGISTIC_C,
                solver="lbfgs",
                l1_ratio=0.0,
                max_iter=2000,
                random_state=seed,
            ),
        }

        for model_name, model in models.items():
            fit_start = perf_counter()

            # 记录是否出现未收敛警告。
            with warnings.catch_warnings(record=True) as caught_warnings:
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

            actual_n_iter = get_actual_n_iter(model)

            records.append({
                "seed": seed,
                "model": model_name,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "n_features": X_train_tfidf.shape[1],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "accuracy_gap": (
                    train_metrics["accuracy"]
                    - val_metrics["accuracy"]
                ),
                "actual_n_iter": actual_n_iter,
                "converged": converged,
                "feature_seconds": feature_seconds,
                "fit_seconds": fit_seconds,
            })

            status = "已收敛" if converged else "未收敛"

            print(f"\n{model_name}")
            print(f"求解器状态: {status}")
            print(f"实际迭代次数: {actual_n_iter}")
            print(
                "训练集: "
                f"Accuracy={train_metrics['accuracy']:.4f}, "
                f"Macro-F1={train_metrics['macro_f1']:.4f}"
            )
            print(
                "验证集: "
                f"Accuracy={val_metrics['accuracy']:.4f}, "
                f"Macro-F1={val_metrics['macro_f1']:.4f}"
            )
            print(
                "训练与验证Accuracy差距: "
                f"{100 * (train_metrics['accuracy'] - val_metrics['accuracy']):.2f}"
                "个百分点"
            )
            print(f"分类器训练耗时: {fit_seconds:.3f}秒")

    results = pd.DataFrame(records)

    summary = (
        results
        .groupby("model", as_index=False)
        .agg(
            runs=("seed", "count"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            accuracy_gap_mean=("accuracy_gap", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .sort_values(
            by="val_macro_f1_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # 同一个随机种子下进行配对比较。
    paired = results.pivot(
        index="seed",
        columns="model",
        values="val_macro_f1",
    ).reset_index()

    paired["svm_minus_lr"] = (
        paired["LinearSVC"]
        - paired["LogisticRegression"]
    )

    svm_wins = int((paired["svm_minus_lr"] > 0).sum())
    lr_wins = int((paired["svm_minus_lr"] < 0).sum())
    ties = int((paired["svm_minus_lr"] == 0).sum())

    mean_difference = paired["svm_minus_lr"].mean()
    std_difference = paired["svm_minus_lr"].std(ddof=1)

    print("\n" + "=" * 60)
    print("=== 五个随机种子的单次结果 ===")

    print(
        results[
            [
                "seed",
                "model",
                "val_accuracy",
                "val_macro_f1",
                "accuracy_gap",
                "actual_n_iter",
                "converged",
                "fit_seconds",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n=== 平均值与标准差 ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n=== 同一随机种子下的配对比较 ===")

    print(
        paired.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n=== 稳定性结论所需数据 ===")
    print(f"LinearSVC胜出次数: {svm_wins}")
    print(f"Logistic Regression胜出次数: {lr_wins}")
    print(f"相同次数: {ties}")
    print(
        "SVM-LR的平均Macro-F1差值: "
        f"{mean_difference:.6f}"
    )
    print(
        "差值的样本标准差: "
        f"{std_difference:.6f}"
    )

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    results_path = (
        output_dir
        / "multi_seed_results.csv"
    )

    summary_path = (
        output_dir
        / "multi_seed_summary.csv"
    )

    paired_path = (
        output_dir
        / "multi_seed_paired_differences.csv"
    )

    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        paired_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 已保存 ===")
    print(results_path)
    print(summary_path)
    print(paired_path)


if __name__ == "__main__":
    main()