# NeuralNetworkMNIST
Neural network train / test for MNIST database

---

Around 90-95% accuracy depending on the constants in the train.cpp file (epochs, learningRate, momentum)

Compile with (-O3 for better runtime):
```
g++ -o train -O3 train.cpp
g++ -o test -O3 test.cpp
```