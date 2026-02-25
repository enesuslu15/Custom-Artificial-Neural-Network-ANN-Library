#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class ActivationFunction {
public:
virtual double activate(double x) = 0;
virtual double derivative(double x) = 0;
};

class Sigmoid : public ActivationFunction {
public:
double activate(double x) override {
return 1.0 / (1.0 + exp(-x));
}

double derivative(double x) override {
double sigmoid = activate(x);
return sigmoid * (1 - sigmoid);
}
};

class Tanh : public ActivationFunction {
public:
double activate(double x) override {
return tanh(x);
}

double derivative(double x) override {
double tanh_x = activate(x);
return 1 - tanh_x * tanh_x;
}
};

class ReLU : public ActivationFunction {
public:
double activate(double x) override {
return std::max(0.0, x);
}

double derivative(double x) override {
return x > 0 ? 1.0 : 0.0;
}
};

class Layer {
public:
std::vector<std::vector<double>> weights;
std::vector<double> biases;
std::vector<double> outputs;
ActivationFunction* activationFunction;

Layer(int inputSize, int outputSize, ActivationFunction* activationFunc) {
activationFunction = activationFunc;
weights.resize(outputSize, std::vector<double>(inputSize));
biases.resize(outputSize);
outputs.resize(outputSize);
initializeWeights();
}

void initializeWeights() {
for (auto& weightRow : weights) {
for (auto& weight : weightRow) {
weight = ((double) std::rand() / RAND_MAX) * 2 - 1;
}
}
for (auto& bias : biases) {
bias = ((double) std::rand() / RAND_MAX) * 2 - 1;
}
}

std::vector<double> forward(const std::vector<double>& inputs) {
for (size_t i = 0; i < weights.size(); ++i) {
double netInput = biases[i];
for (size_t j = 0; j < weights[i].size(); ++j) {
netInput += weights[i][j] * inputs[j];
}
outputs[i] = activationFunction->activate(netInput);
}
return outputs;
}
};

class NeuralNetwork {
public:
std::vector<Layer> layers;

NeuralNetwork(const std::vector<int>& layerSizes, const std::vector<ActivationFunction*>& activations)
{
for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
layers.emplace_back(layerSizes[i], layerSizes[i + 1], activations[i]);
}
}

std::vector<double> predict(const std::vector<double>& inputs) {
std::vector<double> output = inputs;
for (auto& layer : layers) {
output = layer.forward(output);
}
return output;
}
};

int main() {
std::srand(std::time(0));

std::vector<int> layerSizes = {2, 3, 1};
std::vector<ActivationFunction*> activations = {new Sigmoid(), new Tanh()};
NeuralNetwork nn(layerSizes, activations);

std::vector<double> input = {0.5, -0.3};

std::vector<double> output = nn.predict(input);

std::cout << "Output: \n";
for (double val : output) {
std::cout << val << " \n";
}

for (auto activation : activations) {
delete activation;
}

return 0;
} 