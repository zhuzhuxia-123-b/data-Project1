from time import perf_counter
import warnings

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


SVM_C = 1.0

# 每个实验分别与 A 比较，只有一个 TF-IDF 设置发生变化。
EXPERIMENTS = [
    {
        "name": "A_reference",
        "max_features": 5000,
        "ngram_range": (1, 1),
        "sublinear_tf": False,
    },
    {
        "name": "B_vocab_20000",
        "max_features": 20000,
        "ngram_range": (1, 1),
        "sublinear_tf": False,
    },
    {
        "name": "C_add_bigrams",
        "max_features": 5000,
        "ngram_range": (1, 2),
        "sublinear_tf": False,
    },
    {
        "name": "D_sublinear_tf",
        "max_features": 5000,
        "ngram_range": (1, 1),
        "sublinear_tf": True,
    },
]


def main():
    print("=== TF-IDF 对照实验 ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"固定模型: LinearSVC，C={SVM_C}")

    X, y = load_training_data()

    # 所有实验共用这一次划分。
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")

    records = []

    for config in EXPERIMENTS:
        name = config["name"]
        print(f"\n=== {name} ===")
        print(
            f"max_features={config['max_features']}, "
            f"ngram_range={config['ngram_range']}, "
            f"sublinear_tf={config['sublinear_tf']}"
        )

        # 每种配置都重新创建向量化器，
        # 但始终只在同一份训练文本上拟合。
        vectorizer = TfidfVectorizer(
            max_features=config["max_features"],
            ngram_range=config["ngram_range"],
            sublinear_tf=config["sublinear_tf"],
            lowercase=True,
            stop_words=None,
        )

        feature_start = perf_counter()

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)

        feature_seconds = perf_counter() - feature_start

        # 所有实验使用相同的分类器参数。
        model = LinearSVC(
            C=SVM_C,
            dual="auto",
            max_iter=10000,
            random_state=RANDOM_STATE,
        )

        fit_start = perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train_tfidf, y_train)

        fit_seconds = perf_counter() - fit_start

        train_pred = model.predict(X_train_tfidf)
        val_pred = model.predict(X_val_tfidf)

        train_metrics = calculate_metrics(y_train, train_pred)
        val_metrics = calculate_metrics(y_val, val_pred)

        record = {
            "experiment": name,
            "max_features": config["max_features"],
            "ngram_range": str(config["ngram_range"]),
            "sublinear_tf": config["sublinear_tf"],
            "n_features": X_train_tfidf.shape[1],
            "C": SVM_C,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "accuracy_gap": (
                train_metrics["accuracy"] - val_metrics["accuracy"]
            ),
            "feature_seconds": feature_seconds,
            "fit_seconds": fit_seconds,
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "sklearn_version": sklearn.__version__,
        }
        records.append(record)

        print(f"实际特征数: {record['n_features']}")
        print(
            f"训练集: Accuracy={record['train_accuracy']:.4f}, "
            f"Macro-F1={record['train_macro_f1']:.4f}"
        )
        print(
            f"验证集: Accuracy={record['val_accuracy']:.4f}, "
            f"Macro-F1={record['val_macro_f1']:.4f}"
        )
        print(f"特征处理耗时: {feature_seconds:.3f} 秒")
        print(f"分类器训练耗时: {fit_seconds:.3f} 秒")

    results = pd.DataFrame(records)

    # 计算相对 A 的变化，方便进行控制变量分析。
    reference = results.loc[
        results["experiment"] == "A_reference"
    ].iloc[0]

    results["delta_val_accuracy"] = (
        results["val_accuracy"] - reference["val_accuracy"]
    )
    results["delta_val_macro_f1"] = (
        results["val_macro_f1"] - reference["val_macro_f1"]
    )

    # 指标完全相同时，优先实际特征数较少的配置。
    results = results.sort_values(
        by=[
            "val_macro_f1",
            "val_accuracy",
            "n_features",
            "experiment",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    display_columns = [
        "experiment",
        "n_features",
        "train_accuracy",
        "val_accuracy",
        "val_macro_f1",
        "delta_val_macro_f1",
    ]

    print("\n=== TF-IDF 比较结果 ===")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "tfidf_comparison.csv"
    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n本轮结果已保存到: {output_path}")
    print("重复运行本脚本会更新这个结果文件。")


if __name__ == "__main__":
    main()