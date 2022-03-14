import paho.mqtt.client as mqtt
import json
import time
import pickle
import numpy as np
from queue import Queue
from threading import Thread
from InfluxDB import InfluxDB
from sklearn.metrics import r2_score, f1_score


class InferenceWorker(Thread):

    def __init__(self, inference_queue):
        Thread.__init__(self)
        self.inputs, self.key_variables = self.config_parser()
        self.prediction_models = []
        for var in self.key_variables:
            self.prediction_models.append(self.get_model(var, self.ranking_parser(var)))
        self.infer_queue = inference_queue
        self.dbClient = InfluxDB()
        self.AnomalyColumnName = ["perc_error"]
        self.InferenceColumnName = ["actual", "predicted", "performance"]
        self.data_counter = 0
        self.max_data_accumulate = 30
        self.performance_metrics = np.zeros((3, 1))
        self.stored_actual = np.empty((self.max_data_accumulate, 3))
        self.stored_predicted = np.empty((self.max_data_accumulate, 3))

    @staticmethod
    def config_parser():
        with open('config_data.json') as file:
            config_dict = json.load(file)
        inputs = config_dict["inputs"]
        key_variables = config_dict["key_variables"]
        return inputs, key_variables

    @staticmethod
    def ranking_parser(var):
        with open(var + '/ranking.json') as file:
            ranking_dict = json.load(file)
        best_model = ranking_dict["1"]
        return best_model

    @staticmethod
    def get_model(var, model_name):
        print(model_name)
        with open(var + '/' + model_name + '.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    def run(self):
        while True:
            # Get the data from the queue, blocked till data is available
            if self.infer_queue.qsize()>=10:
                data = [self.infer_queue.get() for _ in range(10)]
                self.timestamp = round(time.time() * 1000)
                signal_list = ['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6', 'in_7', 'in_8', 'in_9']
                try:
                    actual_output = np.empty((3, 1))
                    predicted_output = np.empty((3, 1))

                    if not self.inputs:
                        for ind, var in enumerate(self.key_variables):
                            actual_output[ind] = data[signal_list.index(var)]
                            x = data.copy()
                            x.pop(signal_list.index(var))
                            predicted_output[ind] = self.prediction_models[ind].predict(np.array(x).reshape(1, -1))
                            print("predicted_output: ", predicted_output)
                            inferenceData = [
                                {
                                    "measurement": 'Inference',
                                    "tags": {"Signal": self.key_variables[ind]},
                                    "time": self.timestamp,
                                    "fields": dict(zip(self.InferenceColumnName,
                                                       [actual_output[ind], predicted_output[ind],
                                                        self.performance_metrics[ind]]))
                                }]
                            self.dbClient.write_db(inferenceData)  # Check if enrty in ovewritten
                    self.calculate_performance(actual_output.T, predicted_output.T)
                    self.post_process(actual_output, predicted_output)
                    # return None # Why return?

                finally:
                    self.infer_queue.task_done()
            else:
                print("Waiting for all 10 inputs")
                while self.infer_queue.qsize() < 10:
                    time.sleep(0.01)

    def calculate_performance(self, actual_output, predicted_output):
        if self.data_counter < self.max_data_accumulate - 1:
            self.stored_actual[self.data_counter] = actual_output
            self.stored_predicted[self.data_counter] = predicted_output
            self.data_counter += 1
        else:
            np.roll(self.stored_actual, -1, axis=0)
            np.roll(self.stored_predicted, -1, axis=0)
            self.stored_actual[self.data_counter] = actual_output
            self.stored_predicted[self.data_counter] = predicted_output
            for index, _ in enumerate(self.key_variables):
                if len(np.unique(self.stored_actual[:, index])) > 8:
                    self.performance_metrics[index] = 100 * r2_score(self.stored_actual[:, index],
                                                                     self.stored_predicted[:, index])
                else:
                    self.performance_metrics[index] = 100 * f1_score(self.stored_actual[:, index],
                                                                     self.stored_predicted[:, index],
                                                                     average='weighted')

    def post_process(self, actual_output, predicted_output):
        # Enter post process code here
        # Write predicted result to db
        if self.data_counter >= self.max_data_accumulate - 1:
            for ind, _ in enumerate(self.key_variables):
                if len(np.unique(self.stored_actual[:, ind])) > 8:
                    error = ((actual_output[ind] - predicted_output[ind]) / actual_output[ind]) * 100
                    if abs(error) > 20:  # threshold to be added as configurable value
                        data = [
                            {
                                "measurement": 'Anomaly',
                                "tags": {"Signal": self.key_variables[ind]},
                                "time": self.timestamp,
                                "fields": dict(zip(self.AnomalyColumnName, error))
                            }]
                        self.dbClient.write_db(data)
                else:
                    error = actual_output[ind] - predicted_output[ind]
                    if abs(error) >= 1:  # threshold to be added as configurable value
                        data = [
                            {
                                "measurement": 'Anomaly',
                                "tags": {"Signal": self.key_variables[ind]},
                                "time": self.timestamp,
                                "fields": dict(zip(self.AnomalyColumnName, error))
                            }]
                        self.dbClient.write_db(data)


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, userdata2, rc):
    print("Connected with result code " + str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("sensor_data")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    # print(msg.topic + " " + str(msg.payload))
    msg = json.loads(msg.payload.decode())
    inferenceQueue.put((msg["readings"][0]["value"]))


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("127.0.0.1", 1883, 6)

inferenceQueue = Queue()
inference_worker = InferenceWorker(inferenceQueue)
# exit processing when main thread stopped. Need to be checked this or thread.join() is better
inference_worker.daemon = True
inference_worker.start()

client.loop_forever()



