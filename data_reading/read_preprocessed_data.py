import json

import jsonpickle

filename = "data.txt"
# Example of how to read data from file
with open(filename) as json_file:
    data_encoded = json.load(json_file)
    data = jsonpickle.decode(data_encoded)
    print(data[0].features)