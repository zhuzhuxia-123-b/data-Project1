import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from model_config import (
    FINAL_MAX_FEATURES,
    FINAL_NGRAM_RANGE,
    FINAL_SUBLINEAR_TF,
    FINAL_SVM_C,
    FINAL_SVM_MAX_ITER,
    build_final_model,
)

from baseline import (
    DATA_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)
from compare_models import calculate_metrics


LABELS = list(range(10))




def main():
    X, y = load_training_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_final_model()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val, val_pred)

    # 文字版分类报告。
    report_text = classification_report(
        y_val,
        val_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    # 数值版报告，方便筛选召回率较低的类别。
    report_dict = classification_report(
        y_val,
        val_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    summary = (
        "=== 最佳配置验证分析 ===\n"
        f"scikit-learn: {sklearn.__version__}\n"
        f"随机种子: {RANDOM_STATE}\n"
        f"验证集比例: {VALIDATION_SIZE}\n"
        f"TF-IDF: max_features={FINAL_MAX_FEATURES}, "
        f"ngram_range={FINAL_NGRAM_RANGE}, sublinear_tf={FINAL_SUBLINEAR_TF}\n"
        f"模型: LinearSVC(C={FINAL_SVM_C}, dual='auto', max_iter={FINAL_SVM_MAX_ITER})\n"
        f"训练样本数: {len(X_train)}\n"
        f"验证样本数: {len(X_val)}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Macro-F1: {metrics['macro_f1']:.4f}\n"
    )

    print(summary)
    print("=== 各类别分类报告 ===")
    print(report_text)

    class_rows = []
    for label in LABELS:
        item = report_dict[str(label)]
        class_rows.append({
            "class": label,
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1-score"],
            "support": int(item["support"]),
        })

    class_results = pd.DataFrame(class_rows)

    print("=== Recall 最低的三个类别 ===")
    print(
        class_results.sort_values(
            by=["recall", "class"]
        ).head(3).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # 原始计数矩阵，用来找最常见的误分类方向。
    cm = confusion_matrix(y_val, val_pred, labels=LABELS)

    confusion_pairs = []
    for actual in LABELS:
        for predicted in LABELS:
            count = int(cm[actual, predicted])
            if actual != predicted and count > 0:
                confusion_pairs.append((count, actual, predicted))

    confusion_pairs.sort(
        key=lambda item: (-item[0], item[1], item[2])
    )

    print("\n=== 数量最多的五种误分类方向 ===")
    for count, actual, predicted in confusion_pairs[:5]:
        print(
            f"真实类别 {actual} -> 预测类别 {predicted}: "
            f"{count} 条"
        )

    # 保存错误样本，保留它们在原始数据中的位置。
    validation_details = pd.DataFrame({
        "source_index": X_val.index.to_numpy(),
        "true_label": y_val.to_numpy(),
        "predicted_label": val_pred,
        "text": X_val.to_numpy(),
    })

    errors = validation_details.loc[
        validation_details["true_label"]
        != validation_details["predicted_label"]
    ].copy()

    print(f"\n验证集错误样本数: {len(errors)}")
    print("source_index 是原始数据从 0 开始的样本索引。")

    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 图中每行按真实类别归一化，对角线对应 Recall。
    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_val,
        val_pred,
        labels=LABELS,
        display_labels=LABELS,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        colorbar=False,
        ax=ax,
        im_kw={"vmin": 0, "vmax": 1},
    )

    ax.set_title("Validation confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()

    figure_path = output_dir / "best_confusion_matrix.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_path = output_dir / "best_validation_report.txt"
    report_path.write_text(
        summary
        + "\n=== Classification report ===\n"
        + report_text
        + "\n=== Confusion matrix: raw counts ===\n"
        + np.array2string(cm)
        + f"\n\nMisclassified samples: {len(errors)}\n",
        encoding="utf-8",
    )

    errors_path = output_dir / "best_validation_errors.csv"
    errors.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== 已保存 ===")
    print(report_path)
    print(figure_path)
    print(errors_path)
    print("重复运行会更新上述三个文件。")


if __name__ == "__main__":
    main()