import paho.mqtt.client as mqtt
import json
import time
from queue import Queue
from threading import Thread
from InfluxDB import InfluxDB


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
    updaterQueue.put((msg["readings"][0]["value"]))


class DBupdater(Thread):
    """[summary]

    Args:
        Thread ([type]): [description]
    """

    def __init__(self, queue):
        """[summary]

        Args:
            queue ([type]): [description]
        """
        Thread.__init__(self)
        self.queue = queue
        self.dbClient = InfluxDB()
        self.signal_names = ['in_0', 'in_1', 'in_2', 'in_3', 'in_4', 'in_5', 'in_6', 'in_7', 'in_8', 'in_9']

    def run(self):
        """[summary]
        """
        while True:
            # Get the data from the queue, blocked till data is available
            if self.queue.qsize() >= 10:
                try:
                    msg = [self.queue.get() for _ in range(10)]
                    print("Received values", msg)
                    data = [
                        {
                            "measurement": 'DEVICE1',
                            "tags": [],
                            "time": round(time.time() * 1000),
                            "fields": dict(zip(self.signal_names, msg))
                        }
                    ]
                    print(data)
                    self.dbClient.write_db(data)
                    # result = self.dbClient.readDF_db("select * from "+ msg['device']) ######### code to read from db
                    # print(result)
                finally:
                    self.queue.task_done()
            else:
                print("Waiting for all 10 inputs")
                while self.queue.qsize() < 10:
                    time.sleep(0.01)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("127.0.0.1", 1883, 6)

updaterQueue = Queue()
dbUpdater = DBupdater(updaterQueue)
dbUpdater.daemon = True
dbUpdater.start()

# client.subscribe("sensor_data")
# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
