from influxdb import InfluxDBClient, DataFrameClient, exceptions


class InfluxDB(object):
    """Wrapper class for InfluxDB

    Args:
        object ([type]): [description]
    """

    def __init__(self, host='localhost', port=8086) -> None:
        """init constructor

        Args:
            host (str, optional): [description]. Defaults to 'localhost'.
            port (int, optional): [description]. Defaults to 8086.

        Raises:
            TypeError: [description]
            Exception: [description]
            Exception: [description]
        """
        self.user = 'root'
        self.password = 'root'
        self.dbname = 'datadb'
        if not (isinstance(host, str) or isinstance(port, str) or port.isdigit()
                or isinstance(self.dbname, str) or isinstance(self.user, str)
                or isinstance(self.password, str)):
            raise TypeError("Invalid Argument.")
        try:
            self.clientNormal = InfluxDBClient(host, port, self.user, self.password, self.dbname)
            self.clientDF = DataFrameClient(host, port, self.user, self.password, self.dbname)
            self.clientNormal.create_database(self.dbname)
            self.clientNormal.create_retention_policy('awesome_policy', '1d', 3, default=True)
        except exceptions.InfluxDBClientError:
            raise Exception("Can't connect to InluxDB. Request Error")
        except exceptions.InfluxDBServerError:
            raise Exception("Server Error")

    def write_db(self, data):
        """[summary]

        Args:
            data ([type]): [description]
        """
        self.clientNormal.write_points(data, time_precision='ms', protocol='json')

    def readDF_db(self, querry):
        """Returns dataframe

        Args:
            querry ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.clientDF.query(querry)
        return data

    def read_db(self, querry):
        """Returns dictionary

        Args:
            querry ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.clientNormal.query(querry)
        return data

