import zmq
import time
import random
import json
import logging
import numpy as np

# ZeroMQ Context
context = zmq.Context()

# Define the socket using the "Context"
sock = context.socket(zmq.PUB)
sock.bind("tcp://127.0.0.1:5680")


class GenerateSignal(object):
    def __init__(self):
        self.device = 'DEVICE1'
        self.time = time.time()
        self.w1 = 1
        self.w2 = 2
        self.w3 = 3

    def simulate_signals(self):
        """ Generates a new, random signals
        """
        self.time = round(time.time()* 1000)
        signals = []
        for x in range(7):
            signals.append(self.simulate_independent_value())

        signals.append(self.simulate_dependent_continuous_value(signals[0], signals[1], signals[2]))
        signals.append(self.simulate_dependent_continuous_value(signals[1], signals[3], signals[5]))
        signals.append(self.simulate_dependent_categorical_value(signals))

        signal_name = []
        for i in range(10):
            signal_name.append("in_" + str(i))

        return signals, signal_name

    def simulate_independent_value(self):
        return random.uniform(1, 99)

    def simulate_dependent_continuous_value(self, x1, x2, x3):
        return (self.w1 + random.random()) * x1 + (self.w2 + random.random()) * x2 + (self.w3 + random.random()) * x3

    def simulate_dependent_categorical_value(self, signal_list):
        sum_signals = sum(signal_list)
        return 1+int(sum_signals/500 + random.random()*0.2)


ip = GenerateSignal()
print("ZMQ publisher is running")
while True:
    signals, signal_name = ip.simulate_signals()
    print(signals)
    payload = {}
    payload['timestamp'] = ip.time
    payload['device'] = ip.device
    payload['data'] = signals
    payload['signals'] = signal_name
    json_data = json.dumps(payload)
    sock.send_string(ip.device, flags=zmq.SNDMORE)
    sock.send_json(json_data)
    # print(signals)
    time.sleep(0.25)

