from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from baseline import RANDOM_STATE


FINAL_MAX_FEATURES = 20000
FINAL_NGRAM_RANGE = (1, 1)
FINAL_SUBLINEAR_TF = False

FINAL_SVM_C = 1.0
FINAL_SVM_MAX_ITER = 10000


def build_final_model():
    """返回一个尚未训练的最终文本分类Pipeline。"""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=FINAL_MAX_FEATURES,
                ngram_range=FINAL_NGRAM_RANGE,
                sublinear_tf=FINAL_SUBLINEAR_TF,
                lowercase=True,
                stop_words=None,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=FINAL_SVM_C,
                penalty="l2",
                loss="squared_hinge",
                dual="auto",
                max_iter=FINAL_SVM_MAX_ITER,
                random_state=RANDOM_STATE,
            ),
        ),
    ])