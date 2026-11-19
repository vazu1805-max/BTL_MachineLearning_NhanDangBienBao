from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def build_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=25, min_samples_split=4, random_state=42
        ),
        "LinearSVC": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, dual="auto"),
            cv=3
        )
    }
