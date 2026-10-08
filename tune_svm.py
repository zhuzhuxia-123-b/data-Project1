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


C_VALUES = [0.1, 1.0, 10.0]


def main():
    print("=== LinearSVC 参数实验 ===")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"C 候选值: {C_VALUES}")

    X, y = load_training_data()

    # 与前两轮保持相同的数据划分。
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")

    # TF-IDF 设置保持不变，并且只拟合一次。
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        lowercase=True,
        stop_words=None,
    )

    feature_start = perf_counter()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    feature_seconds = perf_counter() - feature_start

    print(f"训练特征矩阵形状: {X_train_tfidf.shape}")
    print(f"验证特征矩阵形状: {X_val_tfidf.shape}")
    print(f"特征处理总耗时: {feature_seconds:.2f} 秒")

    records = []

    for c_value in C_VALUES:
        print(f"\n=== LinearSVC，C={c_value} ===")

        # 每个 C 都重新创建模型，独立训练。
        model = LinearSVC(
            C=c_value,
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
            "model": "LinearSVC",
            "C": c_value,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "accuracy_gap": (
                train_metrics["accuracy"] - val_metrics["accuracy"]
            ),
            "fit_seconds": fit_seconds,
            "random_state": RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "max_features": 5000,
            "ngram_range": "(1, 1)",
            "model_params": str(model.get_params()),
            "sklearn_version": sklearn.__version__,
        }
        records.append(record)

        print(
            f"训练集: Accuracy={record['train_accuracy']:.4f}, "
            f"Macro-F1={record['train_macro_f1']:.4f}"
        )
        print(
            f"验证集: Accuracy={record['val_accuracy']:.4f}, "
            f"Macro-F1={record['val_macro_f1']:.4f}"
        )
        print(
            "训练与验证 Accuracy 差距: "
            f"{record['accuracy_gap'] * 100:.2f} 个百分点"
        )
        print(f"分类器训练耗时: {fit_seconds:.3f} 秒")

    results = pd.DataFrame(records)

    results = results.sort_values(
        by=["val_macro_f1", "val_accuracy", "C"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    display_columns = [
        "C",
        "train_accuracy",
        "train_macro_f1",
        "val_accuracy",
        "val_macro_f1",
        "accuracy_gap",
        "fit_seconds",
    ]

    print("\n=== 参数比较：按验证集 Macro-F1 排序 ===")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best = results.iloc[0]

    print("\n=== 本轮候选值中的最佳配置 ===")
    print(f"C: {best['C']:g}")
    print(f"验证 Accuracy: {best['val_accuracy']:.4f}")
    print(f"验证 Macro-F1: {best['val_macro_f1']:.4f}")
    print("这里只确定当前特征下的候选配置，尚未进行最终训练。")

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "svm_c_results.csv"
    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n本轮结果已保存到: {output_path}")
    print("重复运行本脚本会更新这个结果文件。")


if __name__ == "__main__":
    main()