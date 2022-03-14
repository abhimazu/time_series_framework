# import zmq

# # ZeroMQ Context
# context = zmq.Context()

# # Define the socket using the "Context"
# sock = context.socket(zmq.SUB)

# # Define subscription and messages with prefix to accept.
# # sock.setsockopt_string(zmq.SUBSCRIBE, 'SYMBOL')
# sock.connect("tcp://127.0.0.1:5680")

# while True:
#     message= sock.recv_multipart()
#     print(message)

# -*- coding: utf-8 -*-
# """Tutorial on using the InfluxDB client."""

# import argparse

# from influxdb import InfluxDBClient, DataFrameClient



# def main(host='localhost', port=8086):
#     """Instantiate a connection to the InfluxDB."""
#     user = 'root'
#     password = 'root'
#     dbname = 'example'
#     dbuser = 'smly'
#     dbuser_password = 'my_secret_password'
#     query = 'select * from cpu_load_short;'
#     query_where = 'select Int_value from cpu_load_short where host=$host;'
#     bind_params = {'host': 'server01'}
#     json_body = [
#         {
#             "measurement": "cpu_load_short",
#             "tags": {
#             },
#             "time": 1622798669192,
#             "fields": {
#                 "Float_value": 0.64,
#                 "Int_value": 3,
#                 "String_value": "Text",
#                 "Bool_value": True
#             }
#         }
#     ]

#     client = InfluxDBClient(host, port, user, password, dbname)
#     client2 = DataFrameClient(host, port, user, password, dbname)


#     print("Create database: " + dbname)
#     client.create_database(dbname)

#     # print("Create a retention policy")
#     # client.create_retention_policy('awesome_policy', '3d', 3, default=True)

#     # print("Switch user: " + dbuser)
#     # client.switch_user(dbuser, dbuser_password)

#     for i in range(4):
#         json_body = [
#         {
#             "measurement": "cpu_load_short",
#             "tags": {"signal" : str(i)},
#             "time": 1622798669192,
#             "fields": {
#                 "Float_value": 0.64,
#                 "Int_value": 3,
#                 "String_value": "Text",
#                 "Bool_value": True
#             }
#         }
#         ]

#         # print("Write points: {0}".format(json_body))
#         client.write_points(json_body, time_precision='ms', protocol='json')

#     print("Querying data: " + query)
#     result = client2.query(query)

#     print("Result: {0}".format(result))

#     # print("Querying data: " + query_where)
#     # result = client2.query(query_where, bind_params=bind_params)

#     # print("Result: {0}".format(result))

#     # print("Switch user: " + user)
#     # client.switch_user(user, password)

#     print("Drop database: " + dbname)
#     client.drop_database(dbname)


# def parse_args():
#     """Parse the args."""
#     parser = argparse.ArgumentParser(
#         description='example code to play with InfluxDB')
#     parser.add_argument('--host', type=str, required=False,
#                         default='localhost',
#                         help='hostname of InfluxDB http API')
#     parser.add_argument('--port', type=int, required=False, default=8086,
#                         help='port of InfluxDB http API')
#     return parser.parse_args()


# if __name__ == '__main__':
#     args = parse_args()
#     main(host=args.host, port=args.port)

# from influxdb import InfluxDBClient
# clientNormal = InfluxDBClient('localhost', 8086, 'root', 'password', 'datadb')
# clientNormal.drop_database('datadb')

from influxdb import InfluxDBClient
clientNormal = InfluxDBClient('localhost', 8086, 'root', 'password', 'datadb')


df = clientNormal.query("select * from "+ "Inference")
print(df)