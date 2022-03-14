import numpy as np
import json
import time
from regression_trainer import Regression
from classifier_trainer import Classification
from InfluxDB import InfluxDB


class Federated:

    def __init__(self):
        self.inputs, self.key_variables, self.training_interval = self.config_parser()
        self.min_uniq_thresh, self.regression_info, self.classification_info = self.dev_config_parser()
        self.dbClient = InfluxDB()
        print("Initialized")

    @staticmethod
    def config_parser():
        with open('config_data.json') as file:
            config_dict = json.load(file)
        inputs = config_dict["inputs"]
        key_variables = config_dict["key_variables"]
        training_interval = config_dict["frequency"]
        return inputs, key_variables, training_interval

    @staticmethod
    def dev_config_parser():
        with open('dev_config_data.json') as file:
            config_dict = json.load(file)
        continuous_var_threshold = config_dict["min_unique_for_continuous"]
        regression_config = config_dict["regression_methods"]
        classification_config = config_dict["classification_methods"]
        regression_list = list(regression_config.keys())
        classify_list = list(classification_config.keys())
        regression_method_list = []
        classify_method_list = []
        regress_params = []
        classify_params = []
        regress_default = []
        classify_default = []
        for ind, method in enumerate(regression_list):
            if regression_config[method]["active"]:
                regression_method_list.append(method)
                print(method + " active!")
                default = regression_config[method]["use_default"]
                regress_default.append(default)
                params = regression_config[method]["params"] if not default else {}
                regress_params.append(params)
            else:
                print(method + " regression inactive!")

        for ind, method in enumerate(classify_list):
            if classification_config[method]["active"]:
                classify_method_list.append(method)
                print(method + " active!")
                default = classification_config[method]["use_default"]
                classify_default.append(default)
                params = classification_config[method]["params"] if not default else {}
                classify_params.append(params)
            else:
                print(method + " classification inactive!")

        regression_info = [regression_method_list, regress_default, regress_params]
        classification_info = [classify_method_list, classify_default, classify_params]

        return continuous_var_threshold, regression_info, classification_info

    def get_data(self):
        # df = pd.read_csv('Data_Random.csv')
        # df.columns = ['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6', 'in_7', 'in_8','in_9']
        df = self.dbClient.readDF_db("select * from " + "DEVICE1")
        print(df['DEVICE1'].describe())
        return df['DEVICE1']

    def pre_process(self):
        # df = pd.read_csv('Data_Random.csv')
        # df.columns = ['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6', 'in_7', 'in_8','in_9']
        data = self.get_data()
        # print(data.describe())
        # Add data cleaning code here which filters irregular (mean +- 4 std) data and removes irrelevant inputs
        return data[0:1000]

    def var_type_checker(self, data, variable):
        return len(data[variable].unique()) > self.min_uniq_thresh

    @staticmethod
    def compare_regression_metrics(metrics, variable, methods):
        r2_scores = []
        order_dict = {}
        for method in methods:
            r2_scores.append(metrics[method]["r2_score"])
        sort_index = np.argsort(r2_scores)[::-1]
        for count, ind in enumerate(sort_index):
            order_dict[count + 1] = methods[ind]
        with open(variable + '/ranking.json', 'w') as file:
            json.dump(order_dict, file, ensure_ascii=False, indent=4)

    @staticmethod
    def compare_classification_metrics(metrics, variable, methods):
        f1_scores = []
        order_dict = {}
        for method in methods:
            f1_scores.append(metrics[method]["f1_score"])
        sort_index = np.argsort(f1_scores)[::-1]
        for count, ind in enumerate(sort_index):
            order_dict[count + 1] = methods[ind]
        with open(variable + '/ranking.json', 'w') as file:
            json.dump(order_dict, file, ensure_ascii=False, indent=4)

    def train_models(self):
        train_data = self.pre_process()
        time_conversion_dict = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
            "mon": 2592000
        }
        time_start = time.time()
        # while True:
        #     if not self.inputs:
        #         for var in self.key_variables:
        #             if self.var_type_checker(train_data, var):
        #                 method = Regression()
        #                 train_metrics = method.train(train_data, var, self.regression_info)
        #                 self.compare_regression_metrics(train_metrics, var, list(self.regression_info[0]))
        #             else:
        #                 method = Classification()
        #                 train_metrics = method.train(train_data, var, self.classification_info)
        #                 self.compare_classification_metrics(train_metrics, var, list(self.classification_info[0]))
        #             with open(var + '/metrics.json', 'w') as file:
        #                 json.dump(train_metrics, file, ensure_ascii=False, indent=4)
        #     time.sleep(self.training_interval[0] * time_conversion_dict[self.training_interval[1]] -
        #                (time.time() - time_start))

        if not self.inputs:
            for var in self.key_variables:
                if self.var_type_checker(train_data, var):
                    method = Regression()
                    train_metrics = method.train(train_data, var, self.regression_info)
                    self.compare_regression_metrics(train_metrics, var, list(self.regression_info[0]))
                else:
                    method = Classification()
                    train_metrics = method.train(train_data, var, self.classification_info)
                    self.compare_classification_metrics(train_metrics, var, list(self.classification_info[0]))
                with open(var + '/metrics.json', 'w') as file:
                    json.dump(train_metrics, file, ensure_ascii=False, indent=4)

