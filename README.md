# 当代人工智能实验一：文本分类

本项目使用 TF-IDF 将新闻文本转换为稀疏特征，并比较 Multinomial Naive Bayes、Logistic Regression 和 LinearSVC 三种分类模型。模型选择只使用训练集和验证集，课程提供的无标签测试集仅在最终方案确定后用于生成预测。

## 数据说明

有标签数据包含 7368 条文本，标签范围为 0～9，共 10 个类别。无标签测试集包含 2457 条文本。

数据文件：

- `train_data.csv`
- `test_data_unlabeled.csv`

课程数据不包含在本仓库中。运行代码前，请将以上两个 CSV 文件放到项目根目录。

## 实验环境

- Python 3.12.7
- NumPy 1.26.4
- pandas 2.2.2
- scikit-learn 1.9.0
- Matplotlib 3.9.2
- Random seed: 42

安装依赖：

```bash
python -m pip install -r requirements.txt