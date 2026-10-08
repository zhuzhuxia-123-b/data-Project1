from time import perf_counter
import warnings

import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# 复用上一阶段的数据检查函数和固定设置。
from baseline import (
    DATA_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)


def calculate_metrics(y_true, y_pred):
    """返回 Accuracy 和 Macro-F1。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(10)),
            average="macro",
            zero_division=0,
        ),
    }


def main():
    print("=== 运行环境 ===")
    print(f"pandas: {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")

    X, y = load_training_data()

    # 数据内容、顺序和参数保持不变时，得到与 baseline 相同的划分。
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

    # 与 baseline 使用相同的 TF-IDF 设置。
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        lowercase=True,
        stop_words=None,
    )

    feature_start = perf_counter()

    # 只在训练文本上学习词表和 IDF。
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    feature_seconds = perf_counter() - feature_start

    print("\n=== 共享 TF-IDF 特征 ===")
    print(f"训练特征矩阵形状: {X_train_tfidf.shape}")
    print(f"验证特征矩阵形状: {X_val_tfidf.shape}")
    print(f"特征处理总耗时: {feature_seconds:.2f} 秒")

    models = {
        "MultinomialNB": MultinomialNB(
            alpha=1.0,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "LinearSVC": LinearSVC(
            C=1.0,
            dual="auto",
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
    }

    records = []

    for name, model in models.items():
        print(f"\n=== 训练 {name} ===")

        fit_start = perf_counter()

        # 如果优化没有收敛，就停止，让我们先解决问题，
        # 避免把尚未完成优化的结果直接用于排名。
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train_tfidf, y_train)

        fit_seconds = perf_counter() - fit_start

        train_pred = model.predict(X_train_tfidf)
        val_pred = model.predict(X_val_tfidf)

        train_metrics = calculate_metrics(y_train, train_pred)
        val_metrics = calculate_metrics(y_val, val_pred)

        record = {
            "model": name,
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
        print(f"分类器训练耗时: {fit_seconds:.3f} 秒")

    results = pd.DataFrame(records)

    # 按完整精度排序，显示时才保留四位小数。
    results = results.sort_values(
        by=["val_macro_f1", "val_accuracy", "model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    display_columns = [
        "model",
        "train_accuracy",
        "train_macro_f1",
        "val_accuracy",
        "val_macro_f1",
        "accuracy_gap",
        "fit_seconds",
    ]

    print("\n=== 模型比较：按验证集 Macro-F1 排序 ===")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # 由你运行脚本后，保存本轮真实结果。
    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "model_comparison.csv"
    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n本轮结果已保存到: {output_path}")
    print("重复运行本脚本会更新这个结果文件。")


if __name__ == "__main__":
    main()