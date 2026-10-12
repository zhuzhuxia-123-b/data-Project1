
# 当代人工智能实验一：文本分类

本项目完成《当代人工智能实验课》实验一：文本分类。

实验使用 TF-IDF 将英文新闻文本转换为稀疏特征，并比较以下三种分类模型：

- Multinomial Naive Bayes
- Logistic Regression
- LinearSVC

模型和参数只根据训练集与验证集确定。课程提供的无标签测试集仅在最终方案确定后使用，用于生成 `prediction.csv`。

## 1. 数据说明

有标签数据包含 7368 条文本，标签范围为 0～9，共 10 个类别。各类别样本数基本均衡。

无标签测试集包含 2457 条文本。

运行代码前，需要将以下两个文件放在项目根目录：

```text
train_data.csv
test_data_unlabeled.csv
```

其中：

- `train_data.csv` 包含 `text` 和 `target` 两列；
- `test_data_unlabeled.csv` 只包含 `text` 列；
- 课程数据不上传到 GitHub 仓库。

## 2. 数据划分与评价指标

实验固定使用以下设置：

```text
random_state = 42
validation_size = 0.2
stratify = y
```

7368 条有标签数据被划分为：

- 训练集：5894 条；
- 验证集：1474 条。

评价指标包括：

- Accuracy；
- Macro-F1。

考虑到任务包含 10 个类别，模型选择主要参考 Macro-F1，同时报告 Accuracy。

## 3. 数据隔离原则

为避免数据泄漏，本项目先划分原始文本，再拟合 TF-IDF。

具体规则如下：

1. TF-IDF 只在训练文本上执行 `fit`；
2. 验证集只执行 `transform`，不参与特征词表构建；
3. 测试集不参与模型训练、参数选择或 TF-IDF 拟合；
4. 确定最终模型后，才使用全部有标签数据重新拟合 TF-IDF 和分类器；
5. 最终模型只对测试集执行 `transform` 和 `predict`。

## 4. 实验环境

本项目使用的环境为：

```text
Python          3.12.7
NumPy           1.26.4
pandas          2.2.2
scikit-learn    1.9.0
Matplotlib      3.9.2
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 5. 项目文件

```text
data-Project1/
├── inspect_data.py
├── baseline.py
├── compare_models.py
├── tune_svm.py
├── compare_tfidf.py
├── analyze_best.py
├── svm_loss_curve.py
├── tune_logistic.py
├── check_logistic_c100.py
├── logistic_loss_curve.py
├── generate_prediction.py
├── requirements.txt
├── README.md
└── outputs/
```

各脚本作用如下。

| 脚本                       | 作用                                                   |
| -------------------------- | ------------------------------------------------------ |
| `inspect_data.py`        | 检查数据列名、缺失值、重复文本、标签分布和预测文件格式 |
| `baseline.py`            | 训练5000维TF-IDF与MultinomialNB基线                    |
| `compare_models.py`      | 在相同特征下比较NB、LR和LinearSVC                      |
| `tune_svm.py`            | 比较LinearSVC的不同C参数                               |
| `compare_tfidf.py`       | 比较词表大小、bigram和sublinear TF设置                 |
| `analyze_best.py`        | 分析最佳SVM的分类报告、混淆矩阵和错误样本              |
| `svm_loss_curve.py`      | 计算并绘制LinearSVC的loss检查点曲线                    |
| `tune_logistic.py`       | 比较LR的词表大小以及C=0.1、1、10                       |
| `check_logistic_c100.py` | 补充测试LR的C=100配置                                  |
| `logistic_loss_curve.py` | 计算并绘制LR的log-loss检查点曲线                       |
| `generate_prediction.py` | 使用最终模型重新训练并生成`prediction.csv`           |

## 6. 推荐运行顺序

进入项目目录：

```powershell
cd D:\python\ai\data-Project1
```

依次运行：

```powershell
python inspect_data.py
python baseline.py
python compare_models.py
python tune_svm.py
python compare_tfidf.py
python analyze_best.py
python svm_loss_curve.py
python tune_logistic.py
python check_logistic_c100.py
python logistic_loss_curve.py
```

完成所有训练集和验证集实验、确定最终配置后，再运行：

```powershell
python generate_prediction.py
```

最终预测文件生成在：

```text
outputs/prediction.csv
```

## 7. 初始模型比较

固定使用5000维unigram TF-IDF，三种模型的结果如下。

| 模型                | 训练Accuracy | 验证Accuracy | 验证Macro-F1 |
| ------------------- | -----------: | -----------: | -----------: |
| MultinomialNB       |       0.9454 |       0.8867 |       0.8873 |
| Logistic Regression |       0.9686 |       0.8976 |       0.8985 |
| LinearSVC           |       0.9978 |       0.9186 |       0.9190 |

MultinomialNB训练速度快，作为实验基线。LinearSVC在初始比较中取得最高的验证Macro-F1。

## 8. LinearSVC实验

在5000维unigram TF-IDF下，不同C的结果如下。

|   C | 训练Accuracy | 验证Accuracy | 验证Macro-F1 |
| --: | -----------: | -----------: | -----------: |
| 0.1 |       0.9661 |       0.9003 |       0.9007 |
|   1 |       0.9978 |       0.9186 |       0.9190 |
|  10 |       0.9998 |       0.9118 |       0.9121 |

C=0.1时模型欠拟合。C增至10后训练性能继续提高，但验证性能下降，因此保留C=1。

固定C=1后，TF-IDF实验结果如下。

| 特征设置             | 验证Accuracy | 验证Macro-F1 |
| -------------------- | -----------: | -----------: |
| 5000维unigram        |       0.9186 |       0.9190 |
| 20000维unigram       |       0.9335 |       0.9337 |
| 5000维unigram+bigram |       0.9104 |       0.9107 |
| 5000维sublinear TF   |       0.9186 |       0.9186 |

词表从5000扩大到20000后性能提高。加入bigram后性能下降，因此SVM最终使用20000维unigram TF-IDF。

## 9. Logistic Regression实验

固定20000维unigram TF-IDF，使用L2正则化和lbfgs求解器。

|   C | 实际迭代次数 | 训练Accuracy | 验证Accuracy | 验证Macro-F1 |
| --: | -----------: | -----------: | -----------: | -----------: |
| 0.1 |           18 |       0.9235 |       0.8582 |       0.8600 |
|   1 |           42 |       0.9839 |       0.9138 |       0.9146 |
|  10 |           68 |       0.9998 |       0.9308 |       0.9313 |
| 100 |           64 |       0.9998 |       0.9328 |       0.9333 |

C=0.1时正则化过强。C从10增加到100时，验证Macro-F1只提高0.0020，因此没有继续扩大C的范围。

## 10. Loss曲线说明

`LinearSVC`和`LogisticRegression`不会直接提供完整的逐迭代loss历史。

因此，本项目设置多个`max_iter`检查点，并在每个检查点创建新模型、独立训练，然后根据模型输出计算真实loss。图中的原始数据不进行平滑。

### LinearSVC

LinearSVC使用one-vs-rest squared hinge数据损失。

```text
outputs/svm_loss_history.csv
outputs/svm_loss_curve.png
```

该loss不包含L2正则化项，因此不是模型的完整优化目标。

### Logistic Regression

Logistic Regression使用预测概率计算多分类log loss。

```text
outputs/logistic_loss_history.csv
outputs/logistic_loss_curve.png
```

该loss同样不包含L2正则化项。

这些曲线是不同迭代预算下独立训练得到的检查点曲线，不是单次`fit`内部自动输出的原生逐迭代历史。

## 11. 最终模型

最终模型配置为：

```text
TF-IDF:
    max_features = 20000
    ngram_range = (1, 1)
    sublinear_tf = False

LinearSVC:
    C = 1.0
    penalty = l2
    loss = squared_hinge
    dual = auto
    max_iter = 10000
```

最终模型在当前验证集上的结果为：

```text
Validation Accuracy = 0.9335
Validation Macro-F1 = 0.9337
```

Logistic Regression的最佳验证Macro-F1为0.9333。两个模型表现非常接近，但LinearSVC略高，而且分类器训练速度更快，因此被选作最终模型。

## 12. 输出文件

主要实验输出保存在`outputs`目录：

```text
outputs/
├── model_comparison.csv
├── svm_c_results.csv
├── tfidf_comparison.csv
├── best_validation_report.txt
├── best_confusion_matrix.png
├── svm_loss_history.csv
├── svm_loss_curve.png
├── logistic_c_results.csv
├── logistic_c100_result.csv
├── logistic_loss_history.csv
├── logistic_loss_curve.png
└── prediction.csv
```

`prediction.csv`经过以下检查：

- 共2457行；
- 每行一个0～9之间的整数标签；
- 无表头；
- 无索引；
- 无空白行；
- 顺序与测试集原始顺序一致。

## 13. 复现实验

在数据文件和依赖版本一致的情况下，按照“推荐运行顺序”依次执行脚本，可以重新生成各项实验结果。

由于训练集和验证集划分使用固定随机种子42，同一scikit-learn版本下应得到相同或非常接近的指标。运行时间可能因计算机硬件和系统负载而不同。
