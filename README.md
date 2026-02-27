# 🧠 Custom Artificial Neural Network (ANN) Library

[cite_start]**Technology Stack:** C++ [cite: 332]
[cite_start]**Core Concepts:** Object-Oriented Programming (OOP), Polymorphism, Feedforward Neural Networks, Matrix Math [cite: 342, 367, 391, 403]

## 📌 Project Overview
[cite_start]This project is a lightweight, from-scratch implementation of a Feedforward Artificial Neural Network in C++[cite: 332, 403]. [cite_start]Built with a strong emphasis on Object-Oriented design principles, it abstracts mathematical operations into manageable classes, allowing users to dynamically build networks with custom layer sizes and activation functions[cite: 406, 408].

## ⚙️ Features
* [cite_start]**Polymorphic Activation Functions:** Features an abstract `ActivationFunction` interface with concrete implementations for `Sigmoid`, `Tanh`, and `ReLU` functions, including their mathematical derivatives[cite: 337, 342, 350, 359].
* [cite_start]**Dynamic Layer Configuration:** The `Layer` class handles weight matrices and bias vectors, initializing them with random variables, and computes the forward pass using dot products[cite: 367, 369, 370, 381, 391].
* [cite_start]**Modular Network Builder:** The `NeuralNetwork` class links multiple layers sequentially and feeds the input through the network to generate predictions[cite: 403, 405, 412].

## 🚀 How to Run
1. Compile the code using standard C++ (C++11 or higher):
   `g++ neural_network.cpp -o neural_network`
2. Run the application:
   `./neural_network`
[cite_start]*(The default setup tests a [2, 3, 1] network architecture with dummy input values).* [cite: 422, 426]
