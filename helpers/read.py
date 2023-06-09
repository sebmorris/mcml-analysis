import sqlite3
from helpers.proto import simulation_pb2_v2 as p

class Db:
    def __init__(self, dbpath):
        self.connection = sqlite3.connect(dbpath)
        self.cursor = self.connection.cursor()
    
    def rows(self, type='individual'):
        if type != 'individual' and type != 'parameter':
            raise "type should be 'individual' or 'parameter'"

        self.res = self.cursor.execute('SELECT data FROM archive;')

        for row in self.res:
            if type == 'individual':
                Simulation = p.Simulation()
            else:
                Simulation = p.ParameterSimulation()

            bytesize = Simulation.ParseFromString(row[0])
            yield Simulation
    
    def __del__(self):
        self.connection.close()