from pathlib import Path

import pandas as pd


# 数据文件放在本脚本所在目录，不依赖终端当前所在位置。
DATA_DIR = Path(__file__).resolve().parent


def inspect_table(filename):
    """读取并检查一个带表头的 CSV 文件。"""
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"没有找到文件：{path}")

    df = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(f"文件：{filename}")
    print(f"样本数：{len(df)}")
    print(f"列名：{df.columns.tolist()}")
    print("\n各列数据类型：")
    print(df.dtypes.to_string())
    print("\n各列缺失值数量：")
    print(df.isna().sum().to_string())

    if "text" in df.columns:
        # 检查时不把缺失值转换成字符串 'nan'。
        text = df["text"].astype("string")
        lengths = text.str.len()

        whitespace_only = (
            text.notna()
            & text.str.strip().eq("").fillna(False)
        )

        print(f"\n空字符串或纯空白文本数：{int(whitespace_only.sum())}")
        print(
            "重复文本数（不含缺失值，首次出现不计）："
            f"{int(text.dropna().duplicated().sum())}"
        )
        print("\n文本字符数统计：")
        print(lengths.describe().round(1).to_string())

    return df


def main():
    train = inspect_table("train_data.csv")
    test = inspect_table("test_data_unlabeled.csv")

    print(f"\n{'=' * 60}")
    print("训练数据标签检查")

    if "target" not in train.columns:
        print("没有找到 target 列，请核对老师文件的实际列名。")
    else:
        counts = train["target"].value_counts(
            dropna=False
        ).sort_index()

        distribution = pd.DataFrame({
            "数量": counts,
            "占比(%)": (counts / len(train) * 100).round(2),
        })
        print(distribution.to_string())

        labels = train["target"].dropna().unique().tolist()
        print(f"\n实际标签：{sorted(labels)}")
        print(f"非缺失标签是否恰好为 0～9：{set(labels) == set(range(10))}")

    print(f"\n{'=' * 60}")
    print("训练文本示例：前 3 条，每条最多显示 350 个字符")

    if "text" in train.columns:
        for index, row in train.head(3).iterrows():
            label = row.get("target", "未找到标签列")
            value = row["text"]
            preview = "<缺失>" if pd.isna(value) else str(value)[:350]

            print(f"\n样本 {index}，标签：{label}")
            # repr 让换行、制表符等内容也能看清。
            print(repr(preview))

    print(f"\n{'=' * 60}")
    print("老师的预测示例文件检查")

    example_path = DATA_DIR / "predictions.csv"

    if not example_path.is_file():
        print("没有找到 predictions.csv。")
        return

    with example_path.open(encoding="utf-8-sig", newline="") as file:
        print("前 5 个物理行：")
        for _ in range(5):
            line = file.readline()
            if line == "":
                break
            print(repr(line))

    # 暂按无表头读取，以免把第一条预测误当成列名。
    example = pd.read_csv(
        example_path,
        header=None,
        dtype="string",
        keep_default_na=False,
        skip_blank_lines=False,
    )

    print(f"\n按无表头读取后的形状：{example.shape}")
    print(f"测试样本数：{len(test)}")
    print(f"示例文件行数是否等于测试样本数：{len(example) == len(test)}")
    print("注意：以上只是格式检查，还不能证明示例就是你的最终预测。")


if __name__ == "__main__":
    main()