from _csv import reader

import numpy as np


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        self.W_ih = np.random.rand(hiddennodes, inputnodes)
        self.W_ho = np.random.rand(outputnodes, hiddennodes)

        pass

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def sigmoid_derivative(self, x):
        d = self.sigmoid(x) * (1 - self.sigmoid(x))
        return d

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def train(self, inputs, targets):
        I = self.normalize(inputs)

        H = self.sigmoid((np.matmul(self.W_ih, I)))  # i
        O = self.sigmoid((np.matmul(self.W_ho, H)))

        E_out = targets - O

        E_hidden = np.matmul(self.W_ho.T, E_out)

        delta_W_ho = np.matmul((E_out * self.sigmoid_derivative(O)), H.T)
        self.W_ho = self.W_ho + self.learningrate * delta_W_ho

        delta_W_ih = np.matmul((E_hidden * self.sigmoid_derivative(H)), I.T)
        self.W_ih = self.W_ih + self.learningrate * delta_W_ih
        pass

    # one calculation step of the network
    def think(self, inputs):
        I = self.normalize(inputs)

        H = self.sigmoid((np.matmul(self.W_ih, I)))
        O = self.sigmoid((np.matmul(self.W_ho, H)))
        return O


if __name__ == '__main__':
    input_nodes = 784  # 28*28 pixel
    hidden_nodes = 200  # voodoo magic number
    output_nodes = 10  # numbers from [0:9]

    learning_rate = 0.1
    counter = 0
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("Training...")
    with open('mnist_train.csv', 'r') as read_obj:
       csv_reader = reader(read_obj)
       for x in csv_reader:
           row = x
           T = np.full((10, 1), 0.01)
           x = row[0]
           y = np.asfarray(row[1:]).reshape(784, 1)
           T[int(x), 0] = 0.99
           I = y
           n.train(I,T)

    print("Testing...")
    with open('mnist_test.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for i in csv_reader:
            row = i

            T = np.full((10, 1), 0.01)
            x = row[0]
            y = np.asfarray(row[1:]).reshape(784, 1)

            T[int(x), 0] = 0.99
            I = y
            O = n.think(I)

            result = np.where(O == np.amax(O))
            right_answer = result[0]
            x = row[0]
            if int(right_answer) == int(x):
                counter += 1

    print(f'Test accuracy is {(counter / 10000) * 100}%')