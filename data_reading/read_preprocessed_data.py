import pickle

# Example of how to read data from file
with open('data.obj', 'rb') as json_file:
    data = pickle.load(json_file)
print(data[0].features)