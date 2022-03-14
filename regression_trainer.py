import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


class Regression:

    def __init__(self):
        print("Training Initialized for Continuous Variable:")

    def train(self, data, var, info):
        print(var)
        train_metrics = dict()
        method_list = info[0]
        default_active = info[1]
        params = info[2]
        try:
            if not os.path.exists(var):
                os.makedirs(var)
            for count, method in enumerate(method_list):
                reg = getattr(self, "train_" + method)(default_active[count], params[count])
                train_metrics[method] = self.train_data(data, var, method, reg)
            return train_metrics

        except AttributeError:
            print("The continuous training method '" + method + "' does not exist, kindly check the config files!")

    def train_data(self, data, var, method, reg):
        try:
            print("Training " + var + " with " + method + " regression method.")
            y = np.array(data[var])
            x = np.array(data.drop(columns=[var]))
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            pkl_filename = method + ".pkl"
            with open(var + "/" + pkl_filename, 'wb') as file:
                pickle.dump(reg, file)
            metric_dict = {
                "variance_score": explained_variance_score(y_test, y_pred, multioutput='uniform_average'),
                "r2_score": r2_score(y_test, y_pred),
                "root_mean_sq_err": mean_squared_error(y_test, y_pred, squared=False)
            }
            print("Training of " + var + " with " + method + " regression method complete!")
            return metric_dict

        except ValueError:
            print("Training of " + var + " with " + method + " regression method Incomplete!")
            return {
                "variance_score": 0,
                "r2_score": 0,
                "root_mean_sq_err": 9999
            }

    def train_linear_reg(self, default, param):
        if default:
            reg_trainer = LinearRegression()
        else:
            reg_trainer = LinearRegression(fit_intercept=param["fit_intercept"],
                                           normalize=param["normalize"], positive=param["positive"])
        return reg_trainer

    def train_elastic_net(self, default, param):
        if default:
            reg_trainer = ElasticNet(random_state=0)
        else:
            reg_trainer = ElasticNet(alpha=param["alpha"], l1_ratio=param["l1_ratio"],
                                     fit_intercept=param["fit_intercept"], normalize=param["normalize"],
                                     max_iter=param["max_iter"], tol=param["tol"],
                                     positive=param["positive"], random_state=param["random_state"],
                                     selection=param["selection"])
        return reg_trainer

    def train_kernel_ridge(self, default, param):
        if default:
            reg_trainer = KernelRidge(alpha=0.1, kernel='polynomial', degree=2)
        else:
            reg_trainer = KernelRidge(alpha=param["alpha"], kernel=param["kernel"],
                                      gamma=param["gamma"], degree=param["degree"], coef0=param["coef0"])
        return reg_trainer

    def train_support_vector(self, default, param):
        if default:
            reg_trainer = SVR(kernel='rbf')
        else:
            reg_trainer = SVR(kernel=param["kernel"], degree=param["degree"],
                              gamma=param["gamma"], coef0=param["coef0"],
                              tol=param["tol"], C=param["C"], epsilon=param["epsilon"],
                              shrinking=param["shrinking"], cache_size=param["cache_size"],
                              verbose=param["verbose"], max_iter=param["max_iter"])

        return reg_trainer

    def train_radius_neighbor(self, default, param):
        if default:
            reg_trainer = RadiusNeighborsRegressor(radius=1.0)
        else:
            reg_trainer = RadiusNeighborsRegressor(radius=param["radius"], weights=param["weights"],
                                                   algorithm=param["algorithm"], leaf_size=param["leaf_size"],
                                                   p=param["p"], metric=param["metric"], n_jobs=param["n_jobs"])

        return reg_trainer

    def train_gradient_boost(self, default, param):
        if default:
            reg_trainer = GradientBoostingRegressor(random_state=0)
        else:
            reg_trainer = GradientBoostingRegressor(
                loss=param["loss"], learning_rate=param["learning_rate"],
                n_estimators=param["n_estimators"], subsample=param["subsample"],
                criterion=param["criterion"], min_samples_split=param["min_samples_split"],
                min_samples_leaf=param["min_samples_leaf"], min_weight_fraction_leaf=param["min_weight_fraction_leaf"],
                max_depth=param["max_depth"], min_impurity_decrease=param["min_impurity_decrease"],
                random_state=param["random_state"], max_features=param["max_features"],
                alpha=param["alpha"], verbose=param["verbose"], max_leaf_nodes=param["max_leaf_nodes"],
                validation_fraction=param["validation_fraction"], n_iter_no_change=param["n_iter_no_change"],
                tol=param["tol"], ccp_alpha=param["ccp_alpha"])

        return reg_trainer

    def train_neural_MLP(self, default, param):
        if default:
            reg_trainer = MLPRegressor(random_state=0, max_iter=500)
        else:
            reg_trainer = MLPRegressor(
                hidden_layer_sizes=tuple(param["hidden_layer_sizes"]), activation=param["activation"],
                solver=param["solver"], alpha=param["alpha"], batch_size=param["batch_size"],
                learning_rate=param["learning_rate"], learning_rate_init=param["learning_rate_init"],
                power_t=param["power_t"], max_iter=param["max_iter"], shuffle=param["shuffle"],
                random_state=param["random_state"], tol=param["tol"], verbose=param["verbose"],
                warm_start=param["warm_start"], momentum=param["momentum"],
                nesterovs_momentum=param["nesterovs_momentum"], early_stopping=param["early_stopping"],
                validation_fraction=param["validation_fraction"], beta_1=param["beta_1"],
                beta_2=param["beta_2"], epsilon=param["epsilon"], n_iter_no_change=param["n_iter_no_change"],
                max_fun=param["max_fun"])

        return reg_trainer
