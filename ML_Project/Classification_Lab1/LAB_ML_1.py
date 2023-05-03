# BestCombinationFinder
from textwrap import indent
from typing import Any, Tuple, TypedDict, Union, Optional, Dict
import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    MaxAbsScaler,
    Normalizer,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# type alias for supporting Scalers


class PredictResult:
    # Type Defination for PredictResult

    def __init__(
        self,
        model: Any,
        scaler: Any,
        scaled_attr_names: list[str],
        k_size: Optional[int],
        loss: Any,
        model_args: Optional[Dict[str, Any]],
    ) -> None:
        self.model = model
        self.scaler: str = scaler
        self.scaled_attr_names: list[str] = scaled_attr_names
        self.k_size: Optional[int] = k_size
        self.loss = loss
        self.model_args: Optional[Dict[str, Any]] = model_args
        pass

    def __str__(self, indent=None) -> str:
        dict = {
            "model": type(self.model).__name__,
            "scaler": self.scaler,
            "scaled_attr_names": self.scaled_attr_names,
            "k_size": self.k_size,
            "loss": self.loss,
            "model_args": self.model_args,
        }
        if indent is None:
            return str(dict)
        return json.dumps(
            dict,
            indent=indent,
        )


class BestCombinationFinder:
    """Util for finding best Couple
    Can customize
        - K-Fold size
        - Scaler
        - Encoder
        - Model"""

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Args
            X: DataFrame for independant varibles
            y: DaraFrame for dependant variable
        """
        self.X = X
        self.y = y
        self.k: list[int] = [1]
        self.attributes: list[Any] = []
        self.scalers: list[Any] = [None]
        self.matrix: list[PredictResult] = []

    def set_k_folds(self, k: list[int]):
        """
        set k folding options
        Args
            k: list of fold sizes
        """
        self.k = k
        pass

    def set_scaler(self, attributes: list[Any], scalers: list[Any]):
        """
        set scaling options
        Args
            attribute: list of attribute name that rquired to scaled
            scaler: list of scalrer instance
                ex) [MinMaxScaler()]
        """
        self.attributes = attributes
        self.scalers = scalers

    def evaluate(self, models: list[Any]):
        """
        build coupling matrix
        Args
            models: list of model instance
                ex) [DecisionTree()]
        """
        matrixs: list[PredictResult] = []
        for k in self.k:
            # TODO: split dataset with k-fold
            for scaler in self.scalers:
                X = self.X.copy()
                if scaler != None:
                    X[self.attributes] = scaler.fit_transform(self.X[self.attributes])

                for model in models:
                    # TODO: need to work on k-folded data
                    model = model.fit(X, self.y)
                    predicted = model.predict(self.X)
                    error = mean_squared_error(self.y, predicted)

                    # push result on matrix list
                    args = None
                    name = type(model).__name__
                    if name == "DecisionTreeClassifier":
                        args = {"criterion": model.criterion}
                    elif name == "SVC":
                        args = {
                            "c": model.C,
                            "kernel": model.kernel,
                            "gamma": model.gamma,
                        }
                    matrix = PredictResult(
                        model=model,
                        k_size=k,
                        loss=error,
                        scaler=type(scaler).__name__,
                        scaled_attr_names=self.attributes,
                        model_args=args,
                    )
                    matrixs.append(matrix)
        self.matrix = matrixs
        pass

    def get_best_couple(self) -> PredictResult:
        """
        get best couple data
        """
        ret = self.matrix[0]
        for item in self.matrix:
            if ret.loss < item.loss:
                ret = item
        return ret


feature_names = [
    "Sample code number",
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "Class",
]


def load_data() -> Tuple[pd.DataFrame, pd.DatetimeTZDtype]:
    """
    load local data and preprocessing

    origin data format
    #  Attribute                     Domain
    -- -----------------------------------------
    1. Sample code number            id number
    2. Clump Thickness               1 - 10
    3. Uniformity of Cell Size       1 - 10
    4. Uniformity of Cell Shape      1 - 10
    5. Marginal Adhesion             1 - 10
    6. Single Epithelial Cell Size   1 - 10
    7. Bare Nuclei                   1 - 10
    8. Bland Chromatin               1 - 10
    9. Normal Nucleoli               1 - 10
    10. Mitoses                      1 - 10
    11. Class:                       2(benign), 4(malignant)
    """

    df = pd.read_csv("./breast-cancer-wisconsin.data", header=None)
    df.columns = feature_names
    df.drop([feature_names[0]], axis=1, inplace=True)
    # drop ?
    df = df[df["Bare Nuclei"] != "?"]
    df["Bare Nuclei"] = pd.to_numeric(df["Bare Nuclei"])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


if __name__ == "__main__":

    X, y = load_data()
    finder = BestCombinationFinder(X, y)
    # finder.set_scaler(, [])
    # finder.set_encoder()
    finder.set_scaler(
        feature_names[1:-1],
        [
            MinMaxScaler(),
            RobustScaler(),
            # StandardScaler(),
            # MaxAbsScaler(),
            # Normalizer(),
        ],
    )

    svm_list: list[SVC] = []
    for kernel in ["linear", "poly", "rbf", "sigmoid", "precomputed"]:
        for c in [1, 2, 3, 4]:
            for gamma in ["scale", "auto"]:
                svm = SVC(kernel=kernel, C=c, gamma=gamma)
                svm_list.append(svm)

    finder.evaluate(
        [
            DecisionTreeClassifier(criterion="gini"),
            DecisionTreeClassifier(criterion="entropy"),
            LogisticRegression(),
        ]
        + svm_list
    )
    import json

    print(finder.get_best_couple())
    for item in finder.matrix:
        print(item)
    # print(json.dumps(json.loads(str(finder.matrix)), indent=4))
