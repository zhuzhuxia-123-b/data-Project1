from time import perf_counter
import warnings

import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

from baseline import DATA_DIR, load_training_data
from model_config import build_final_model


def load_test_texts():
    """读取测试文本，并保持文件中的原始样本顺序。"""
    test_path = DATA_DIR / "test_data_unlabeled.csv"
    test_df = pd.read_csv(test_path)

    if "text" not in test_df.columns:
        raise ValueError("测试文件缺少 text 列。")

    if "target" in test_df.columns:
        raise ValueError("读到的文件包含 target，请确认使用了无标签测试文件。")

    if test_df.empty:
        raise ValueError("测试文件没有样本。")

    if test_df["text"].isna().any():
        raise ValueError("测试文本存在缺失值，请先检查。")

    test_texts = test_df["text"].astype(str)

    if test_texts.str.strip().eq("").any():
        raise ValueError("测试文本存在空白内容，请先检查。")

    # 不排序、不删除样本，确保预测顺序与原始文件一致。
    return test_texts


def verify_prediction_file(output_path, predictions, expected_rows):
    """重新读取已保存文件，检查提交格式和预测顺序。"""
    lines = output_path.read_text(encoding="utf-8").splitlines()

    if len(lines) != expected_rows:
        raise ValueError(
            f"预测文件有 {len(lines)} 行，"
            f"但测试集有 {expected_rows} 条样本。"
        )

    allowed_labels = {str(label) for label in range(10)}

    if any(line not in allowed_labels for line in lines):
        raise ValueError(
            "预测文件格式错误：每行必须只有一个 0～9 的整数标签。"
        )

    saved_predictions = [int(line) for line in lines]

    if saved_predictions != predictions.tolist():
        raise ValueError("保存后的预测值或顺序与模型输出不一致。")


def main():
    print("=== 最终训练与测试集预测 ===")
    print(f"scikit-learn: {sklearn.__version__}")

    # 本阶段使用全部有标签数据，不再重新划分验证集。
    X_all, y_all = load_training_data()
    X_test = load_test_texts()

    print(f"完整有标签数据样本数: {len(X_all)}")
    print(f"测试样本数: {len(X_test)}")

    # 创建全新的 Pipeline，复用已经确定的配置。
    model = build_final_model()

    print("\n开始使用全部有标签数据重新训练……")
    training_start = perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_all, y_all)

    training_seconds = perf_counter() - training_start

    vocabulary_size = len(
        model.named_steps["tfidf"].vocabulary_
    )

    print(f"最终词表大小: {vocabulary_size}")
    print(f"TF-IDF 拟合及分类器训练耗时: {training_seconds:.3f} 秒")

    print("\n开始预测测试集……")
    prediction_start = perf_counter()

    # Pipeline 只对测试文本做 transform 和 predict，
    # 不会在测试文本上重新拟合 TF-IDF。
    predictions = model.predict(X_test)

    prediction_seconds = perf_counter() - prediction_start

    if len(predictions) != len(X_test):
        raise ValueError("预测数量与测试样本数不一致。")

    if not pd.api.types.is_integer_dtype(predictions.dtype):
        raise ValueError("模型输出不是整数标签，请检查。")

    if not pd.Series(predictions).isin(range(10)).all():
        raise ValueError("模型输出包含 0～9 以外的标签。")

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 注意最终提交文件名为 prediction.csv。
    output_path = output_dir / "prediction.csv"

    pd.DataFrame({"target": predictions}).to_csv(
        output_path,
        index=False,
        header=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    # 保存后重新检查实际文件，而不只检查内存中的预测。
    verify_prediction_file(
        output_path=output_path,
        predictions=predictions,
        expected_rows=len(X_test),
    )

    print(f"测试集预测耗时: {prediction_seconds:.3f} 秒")
    print("\n=== 预测文件检查通过 ===")
    print(f"文件位置: {output_path}")
    print(f"预测行数: {len(predictions)}")
    print("每行一个整数标签，范围为 0～9")
    print("无表头、无索引、无空白行")
    print("保存顺序与测试数据原始顺序一致")
    print("重复运行本脚本会更新这个预测文件。")


if __name__ == "__main__":
    main()