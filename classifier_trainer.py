import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


class Classification:

    def __init__(self):
        print("Training Initialized for Continuous Variable:")

    def train(self, data, var, info):
        print(var)
        train_metrics = dict()
        method_list = info[0]
        default_active = info[1]
        params = info[2]
        if not os.path.exists(var):
            os.makedirs(var)
        try:
            for count, method in enumerate(method_list):
                cls = getattr(self, "train_" + method)(default_active[count], params[count])
                train_metrics[method] = self.train_data(data, var, method, cls)
            return train_metrics
        except AttributeError:
            print("The continuous training method '" + method + "' does not exist, kindly check the config files!")

    @staticmethod
    def train_data(self, data, var, method, cls):
        try:
            print("Training " + var + " with " + method + " classification method.")
            y = np.array(data[var])
            x = np.array(data.drop(columns=[var]))
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            cls.fit(x_train, y_train)
            y_pred = cls.predict(x_test)
            pkl_filename = method + ".pkl"
            with open(var + "/" + pkl_filename, 'wb') as file:
                pickle.dump(cls, file)
            plot_confusion_matrix(cls, x_test, y_test)
            plt.title(var + ": Confusion Matrix by " + method + " classification")
            plt.ylabel('Actual Classes')
            plt.xlabel('Predicted Classes')
            plt.savefig(var + "/" + method + "_cm.jpg")
            metric_dict = {
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
            print("Training of " + var + " with " + method + " classification method complete!")
            return metric_dict

        except:
            print("Training of " + var + " with " + method + " classification method incomplete!")
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0
            }

    @staticmethod
    def train_logistic_reg(self, default, param):
        if default:
            cls_trainer = LogisticRegression(random_state=0)
        else:
            cls_trainer = LogisticRegression(
                penalty=param["penalty"], dual=param["dual"], tol=param["tol"],
                C=param["C"], fit_intercept=param["fit_intercept"],
                intercept_scaling=param["intercept_scaling"], class_weight=param["class_weight"],
                random_state=param["random_state"], solver=param["solver"],
                max_iter=param["max_iter"], multi_class=param["multi_class"],
                verbose=param["verbose"], warm_start=param["warm_start"],
                n_jobs=param["n_jobs"], l1_ratio=param["l1_ratio"])

        return cls_trainer

    @staticmethod
    def train_ridge(self, default, param):
        if default:
            cls_trainer = RidgeClassifier(random_state=0)
        else:
            cls_trainer = RidgeClassifier(
                alpha=param["alpha"], fit_intercept=param["fit_intercept"],
                normalize=param["normalize"], max_iter=param["max_iter"],
                tol=param["tol"], class_weight=param["class_weight"],
                random_state=param["random_state"], solver=param["solver"])

        return cls_trainer

    @staticmethod
    def train_naive_bayes(self, default, param):
        if default:
            cls_trainer = MultinomialNB()
        else:
            cls_trainer = MultinomialNB(alpha=param["alpha"], fit_prior=param["fit_prior"],
                                        class_prior=param["class_prior"])

        return cls_trainer

    @staticmethod
    def train_KNN(self, default, param):
        if default:
            cls_trainer = KNeighborsClassifier()
        else:
            cls_trainer = KNeighborsClassifier(
                n_neighbors=param["n_neighbors"], weights=param["weights"],
                algorithm=param["algorithm"], max_iter=param["max_iter"],
                leaf_size=param["leaf_size"], p=param["p"], metric=param["metric"],
                metric_params=param["metric_params"], n_jobs=param["n_jobs"]
            )

        return cls_trainer

    @staticmethod
    def train_random_forest(self, default, param):
        if default:
            cls_trainer = RandomForestClassifier(random_state=0)
        else:
            cls_trainer = RandomForestClassifier(
                n_estimators=param["n_estimators"], criterion=param["criterion"], max_depth=param["max_depth"],
                min_samples_split=param["min_samples_split"], min_samples_leaf=param["min_samples_leaf"],
                min_weight_fraction_leaf=param["min_weight_fraction_leaf"], max_features=param["max_features"],
                random_state=param["random_state"], max_leaf_nodes=param["max_leaf_nodes"],
                min_impurity_decrease=param["min_impurity_decrease"], bootstrap=param["bootstrap"],
                oob_score=param["oob_score"], n_jobs=param["n_jobs"], verbose=param["verbose"],
                warm_start=param["warm_start"], class_weight=param["class_weight"], ccp_alpha=param["ccp_alpha"],
                max_samples=param["max_samples"])

        return cls_trainer

    @staticmethod
    def train_decision_tree(self, default, param):
        if default:
            cls_trainer = DecisionTreeClassifier(random_state=0)
        else:
            cls_trainer = DecisionTreeClassifier(
                criterion=param["criterion"], splitter=param["splitter"], max_depth=param["max_depth"],
                min_samples_split=param["min_samples_split"], min_samples_leaf=param["min_samples_leaf"],
                min_weight_fraction_leaf=param["min_weight_fraction_leaf"], max_features=param["max_features"],
                random_state=param["random_state"], max_leaf_nodes=param["max_leaf_nodes"],
                min_impurity_decrease=param["min_impurity_decrease"], class_weight=param["class_weight"],
                ccp_alpha=param["ccp_alpha"]
            )

        return cls_trainer

    @staticmethod
    def train_support_vector(self, default, param):
        if default:
            cls_trainer = LinearSVC(random_state=0)
        else:
            cls_trainer = LinearSVC(
                C=param["C"], kernel=param["kernel"], degree=param["degree"], gamma=param["gamma"],
                coef0=param["coef0"], shrinking=param["shrinking"], probability=param["probability"], tol=param["tol"],
                cache_size=param["cache_size"], class_weight=param["class_weight"], verbose=param["verbose"],
                max_iter=param["max_iter"], decision_function_shape=param["decision_function_shape"],
                break_ties=param["break_ties"], random_state=param["random_state"]
            )

        return cls_trainer

    @staticmethod
    def train_neural_MLP(self, default, param):
        if default:
            cls_trainer = MLPClassifier(random_state=0, max_iter=300)
        else:
            cls_trainer = MLPClassifier(
                hidden_layer_sizes=tuple(param["hidden_layer_sizes"]), activation=param["activation"],
                solver=param["solver"],
                alpha=param["alpha"], batch_size=param["batch_size"], learning_rate=param["learning_rate"],
                learning_rate_init=param["learning_rate_init"], power_t=param["power_t"], max_iter=param["max_iter"],
                shuffle=param["shuffle"], random_state=param["random_state"], tol=param["tol"],
                verbose=param["verbose"], warm_start=param["warm_start"], momentum=param["momentum"],
                nesterovs_momentum=param["nesterovs_momentum"], early_stopping=param["early_stopping"],
                validation_fraction=param["validation_fraction"], beta_1=param["beta_1"], beta_2=param["beta_2"],
                epsilon=param["epsilon"], n_iter_no_change=param["n_iter_no_change"], max_fun=param["max_fun"]
            )

        return cls_trainer
