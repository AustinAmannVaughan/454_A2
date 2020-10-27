#Austin Amann Vaughan Assignment #2 code
#10101667
from random import randrange
from random import random
import csv
from math import exp

# The following functions are used to input the glass data and process it so that it can be used by the neural network
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
def convertColumn(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
def convertColumnFloat(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats



def cross_validation_split(dataset, n_splits):
	dataset_split = list()
	dataset_copy = list(dataset)
	splits_size = int(len(dataset) / n_splits)
	for i in range(n_splits):
		splits = list()
		while len(splits) < splits_size:
			index = randrange(len(dataset_copy))
			splits.append(dataset_copy.pop(index))
		dataset_split.append(splits)
	return dataset_split
#END OF PREPROCESSING

# Used to calculate the positive predictions
def accuracyCalc(test, result):
	posRes = 0
	for i in range(len(test)):
		if test[i] == result[i]:
			posRes += 1
	return posRes / float(len(test)) * 100.0

# Determines the score of the algorithm
def scoreResults(dataset, algorithm, n_splits, *args):
	splitss = cross_validation_split(dataset, n_splits)
	scores = list()
	for splits in splitss:
		trainingData = list(splitss)
		trainingData.remove(splits)
		trainingData = sum(trainingData, [])
		testData = list()
		for row in splits:
			row_copy = list(row)
			testData.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(trainingData, testData, *args)
		actual = [row[-1] for row in splits]
		accuracy = accuracyCalc(actual, predicted)
		scores.append(accuracy)
	return scores

# activates neurons in the network
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# sigmoid transfer function for the neurons
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagation for ANN
def forward_propagate(ANN, row):
	inputs = row
	for layer in ANN:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# derivitive of output of the neuroms
def transfer_derivative(output):
	return output * (1.0 - output)

# calculate the error in the back propagation
def backward_propagate_error(ANN, expected):
	for i in reversed(range(len(ANN))):
		layer = ANN[i]
		errors = list()
		if i != len(ANN)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in ANN[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# update the ANNs weights with calculated error
def update_weights(ANN, row, learn):
	for i in range(len(ANN)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in ANN[i - 1]]
		for neuron in ANN[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += learn * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += learn * neuron['delta']

# Train a network for provided number of iterations
def NetworkTraining(ANN, train, learn, iterations, olayers):
	for epoch in range(iterations):
		for row in train:
			outputs = forward_propagate(ANN, row)
			expected = [0 for i in range(olayers)]
			expected[row[-1]] = 1
			backward_propagate_error(ANN, expected)
			update_weights(ANN, row, learn)
	print("Final Weights Vectors: %s"%ANN[1])

# Initialize the ANN
def network_init(ilayers, hlayers, olayers):
	ANN = list()
	hiddenLayer= [{'weights':[random() for i in range(ilayers
                                                        + 1)]} for i in range(hlayers)]
	ANN.append(hiddenLayer)
	output_layer = [{'weights':[random() for i in range(hlayers + 1)]} for i in range(olayers)]
	ANN.append(output_layer)
	return ANN

# Make predictions using the ANN
def predict(ANN, row):
	outputs = forward_propagate(ANN, row)
	return outputs.index(max(outputs))

# Backpropagation wih gradient decent
def back_propagation(train, test, learn, iterations, hlayers):
        predictions = list()
        real_predictions = list()
        ilayers = len(train[0]) - 1
        print("INPUT LAYERS",ilayers)
        olayers = len(set([row[-1] for row in train]))
        print("OUTPUT LAYERS",olayers)
        print("HIDDEN LAYERS",hlayers)
        ANN = network_init(ilayers, hlayers, olayers)
        NetworkTraining(ANN, train, learn, iterations, olayers)
        for row in test:
            prediction = predict(ANN, row)
            pred_real = find_label(prediction)
            predictions.append(prediction)
            real_predictions.append(pred_real)
            
        with open('predictions.csv','w') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerow(real_predictions)
        return(predictions)
#finds the actual label for the predictions the ANN finds as numbers
def find_label(pred):
    if pred==1:
        return("building_windows_float_processed")
    elif pred==2:
        return("building_windows_non_float_processed")
    elif pred==3:
        return("vehicle_windows_float_processed")
    elif pred==4:
        return("vehicle_windows_non_float_processed")
    elif pred==5:
        return("containers")
    elif pred==6:
        return("tableware")
    elif pred==7:
        return("headlamps")
        
# load and prepare data
filename = 'GlassData.csv'
dataset = load_csv(filename)
dataset.pop(0)
for i in range(len(dataset[0])-1):
	convertColumnFloat(dataset, i)

convertColumn(dataset, len(dataset[0])-1)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate performance of the backpropagation ANN
#number of iterations that chosen iterations is run
n_splits = 2
#learning rate
learn = 0.3
#chosen # of iterations to be run
iterations = 500
# of hidden layers
hlayers = 5
scores = scoreResults(dataset, back_propagation, n_splits, learn, iterations, hlayers)
#Print the % of true and false predicitons
print('True: %s' % scores)
false = list()
false = scores
for i in range(len(scores)):
    false[i]=100.0-scores[i]
print('False: %s'% false)
