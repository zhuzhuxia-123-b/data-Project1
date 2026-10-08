from pathlib import Path
from time import perf_counter

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

# 根据脚本位置寻找数据，不在代码中写死电脑的绝对路径。
DATA_DIR = Path(__file__).resolve().parent


def load_training_data():
    """读取有标签数据，检查后返回文本和标签。"""
    data_path = DATA_DIR / "train_data.csv"
    df = pd.read_csv(data_path)

    required_columns = {"text", "target"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"数据必须包含 text 和 target 两列，实际列名为：{df.columns.tolist()}"
        )

    if df[["text", "target"]].isna().any().any():
        raise ValueError("发现缺失文本或标签，请先检查数据。")

    X = df["text"].astype(str)

    if X.str.strip().eq("").any():
        raise ValueError("发现空白文本，请先检查数据。")

    # 先检查标签，再转换为整数，避免把非法小数标签直接截断。
    raw_y = df["target"]

    if not raw_y.isin(range(10)).all():
        raise ValueError("发现不属于整数 0～9 的标签。")

    y = raw_y.astype(int)

    if set(y.unique()) != set(range(10)):
        raise ValueError("数据没有完整包含 0～9 十个类别。")

    return X, y


def print_metrics(name, y_true, y_pred):
    """打印整体准确率和十个类别的平均 F1。"""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(10)),
        average="macro",
        zero_division=0,
    )

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")


def main():
    print("=== 运行环境 ===")
    print(f"pandas: {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子: {RANDOM_STATE}")

    X, y = load_training_data()

    # 必须先划分原始文本，然后才能拟合 TF-IDF。
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n=== 数据划分 ===")
    print(f"有标签数据总数: {len(X)}")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")

    class_counts = pd.DataFrame({
        "训练集": y_train.value_counts().sort_index(),
        "验证集": y_val.value_counts().sort_index(),
    })
    class_counts.index.name = "类别"

    print("\n各类别样本数:")
    print(class_counts.to_string())

    # Pipeline 将 TF-IDF 和分类器连接起来。
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 1),
                lowercase=True,
                stop_words=None,
            ),
        ),
        (
            "classifier",
            MultinomialNB(alpha=1.0),
        ),
    ])

    print("\n=== 开始训练 baseline ===")
    print("特征: TF-IDF，unigram，max_features=5000")
    print("模型: MultinomialNB，alpha=1.0")

    start_time = perf_counter()

    # 这里只传入训练集，验证集不参与拟合。
    model.fit(X_train, y_train)

    training_seconds = perf_counter() - start_time

    vocabulary_size = len(model.named_steps["tfidf"].vocabulary_)
    print(f"实际词表大小: {vocabulary_size}")
    print(f"特征拟合及模型训练耗时: {training_seconds:.2f} 秒")

    # 训练集指标用于辅助观察拟合情况。
    train_predictions = model.predict(X_train)

    # 验证集指标用于评价当前方案。
    val_predictions = model.predict(X_val)

    print_metrics("=== 训练集表现 ===", y_train, train_predictions)
    print_metrics("=== 验证集表现 ===", y_val, val_predictions)

    print("\nBaseline 完成，请保留本次配置和输出。")


if __name__ == "__main__":
    main()