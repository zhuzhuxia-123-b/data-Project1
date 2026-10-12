from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from baseline import (
    DATA_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)
from compare_models import calculate_metrics


SVM_C = 1.0
MAX_FEATURES = 20000


def get_actual_n_iter(model):
    """返回LinearSVC实际使用的迭代次数。"""
    n_iter_array = np.asarray(model.n_iter_)
    return int(n_iter_array.max())


def build_feature_extractors():
    """创建三个受控的特征实验。"""

    word_20000 = TfidfVectorizer(
        analyzer="word",
        max_features=20000,
        ngram_range=(1, 1),
        lowercase=True,
        sublinear_tf=False,
        stop_words=None,
    )

    char_20000 = TfidfVectorizer(
        analyzer="char_wb",
        max_features=20000,
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=False,
    )

    # word和char各占10000维，总上限仍为20000。
    hybrid_20000 = Pipeline([
        (
            "union",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        max_features=10000,
                        ngram_range=(1, 1),
                        lowercase=True,
                        sublinear_tf=False,
                        stop_words=None,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        max_features=10000,
                        ngram_range=(3, 5),
                        lowercase=True,
                        sublinear_tf=False,
                    ),
                ),
            ]),
        ),
        # 两组TF-IDF拼接后重新归一化，
        # 避免组合特征整体尺度发生变化。
        (
            "normalize",
            Normalizer(
                norm="l2",
                copy=False,
            ),
        ),
    ])

    return {
        "A_word_20000": {
            "description": "word unigram, 20000 features",
            "extractor": word_20000,
        },
        "B_char_wb_20000": {
            "description": "char_wb (3,5), 20000 features",
            "extractor": char_20000,
        },
        "C_word_char_10000_10000": {
            "description": (
                "word unigram 10000 + "
                "char_wb (3,5) 10000"
            ),
            "extractor": hybrid_20000,
        },
    }


def main():
    print("=== 字符级TF-IDF创新实验 ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"验证集比例: {VALIDATION_SIZE}")
    print(f"固定模型: LinearSVC(C={SVM_C})")
    print("各实验最大特征总数: 20000")
    print("本实验不读取测试集。")

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

    experiments = build_feature_extractors()
    records = []

    for experiment_name, config in experiments.items():
        print("\n" + "=" * 60)
        print(f"实验: {experiment_name}")
        print(f"特征: {config['description']}")

        extractor = config["extractor"]

        feature_start = perf_counter()

        # 每个特征方案只在训练文本上拟合。
        X_train_features = extractor.fit_transform(
            X_train
        )
        X_val_features = extractor.transform(
            X_val
        )

        feature_seconds = perf_counter() - feature_start

        actual_features = X_train_features.shape[1]

        # 平均每条文本中非零特征的数量，
        # 用于辅助比较特征矩阵复杂度。
        average_nonzero = (
            X_train_features.nnz
            / X_train_features.shape[0]
        )

        print(
            "训练特征矩阵形状: "
            f"{X_train_features.shape}"
        )
        print(
            "验证特征矩阵形状: "
            f"{X_val_features.shape}"
        )
        print(
            "平均每条训练文本的非零特征数: "
            f"{average_nonzero:.2f}"
        )
        print(
            f"特征处理耗时: "
            f"{feature_seconds:.3f}秒"
        )

        model = LinearSVC(
            C=SVM_C,
            penalty="l2",
            loss="squared_hinge",
            dual="auto",
            max_iter=10000,
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
                X_train_features,
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
        actual_n_iter = get_actual_n_iter(model)

        train_predictions = model.predict(
            X_train_features
        )
        val_predictions = model.predict(
            X_val_features
        )

        train_metrics = calculate_metrics(
            y_train,
            train_predictions,
        )
        val_metrics = calculate_metrics(
            y_val,
            val_predictions,
        )

        records.append({
            "experiment": experiment_name,
            "description": config["description"],
            "actual_features": actual_features,
            "average_nonzero_features": average_nonzero,
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

        print(
            f"分类器训练耗时: "
            f"{fit_seconds:.3f}秒"
        )

    results = pd.DataFrame(records)

    baseline_macro_f1 = results.loc[
        results["experiment"] == "A_word_20000",
        "val_macro_f1",
    ].iloc[0]

    results["delta_val_macro_f1"] = (
        results["val_macro_f1"]
        - baseline_macro_f1
    )

    results = results.sort_values(
        by="val_macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("=== 字符特征比较结果 ===")

    display_columns = [
        "experiment",
        "actual_features",
        "average_nonzero_features",
        "train_accuracy",
        "val_accuracy",
        "val_macro_f1",
        "delta_val_macro_f1",
        "accuracy_gap",
        "feature_seconds",
        "fit_seconds",
        "converged",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    best = results.iloc[0]

    print("\n=== 当前划分中的最佳特征方案 ===")
    print(f"实验: {best['experiment']}")
    print(f"说明: {best['description']}")
    print(
        "验证Accuracy: "
        f"{best['val_accuracy']:.4f}"
    )
    print(
        "验证Macro-F1: "
        f"{best['val_macro_f1']:.4f}"
    )
    print(
        "相对词级基线的Macro-F1变化: "
        f"{best['delta_val_macro_f1']:+.4f}"
    )

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / "char_feature_comparison.csv"
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 已保存 ===")
    print(output_path)
    print(
        "这里只比较训练集和验证集，"
        "尚未重新训练最终模型。"
    )


if __name__ == "__main__":
    main()
    