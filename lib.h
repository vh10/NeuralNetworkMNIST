#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iomanip>

using namespace std;

const string testingImageFile = "data/t10k-images.idx3-ubyte";
const string testingLabelFile = "data/t10k-labels.idx1-ubyte";
const string trainingImageFile = "data/train-images.idx3-ubyte";
const string trainingLabelFile = "data/train-labels.idx1-ubyte";
const string networkFile = "neuralNetworkSave";

const int nrTests = 10000;
const int nrTrain = 60000;

const int width = 28;
const int height = 28;

const int n1 = width * height;
const int n2 = 128; 
const int n3 = 10;

double w1[n1 + 1][n2 + 1], out1[n1 + 1];
double w2[n2 + 1][n3 + 1], in2[n2 + 1], out2[n2 + 1];
double in3[n3 + 1], out3[n3 + 1];

double expected[n3 + 1];

int image[width + 1][height + 1];

ifstream imageFile;
ifstream labelFile;

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void feedForward() {
    for (int i = 1; i <= n2; ++i)
		in2[i] = 0.0;
    for (int i = 1; i <= n3; ++i)
		in3[i] = 0.0;

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j)
            in2[j] += out1[i] * w1[i][j];
	}

    for (int i = 1; i <= n2; ++i)
		out2[i] = sigmoid(in2[i]);

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j)
            in3[j] += out2[i] * w2[i][j];
	}

    for (int i = 1; i <= n3; ++i)
		out3[i] = sigmoid(in3[i]);
}

double resultError(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i)
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);

    res *= 0.5;
    return res;
}

int readImage() {
    char x;
    for (int i = 1; i <= width; ++i) {
        for (int j = 1; j <= height; ++j) {
            imageFile.read(&x, sizeof(char));
            image[i][j] = x ? 1 : 0;

            int pos = i + (j - 1) * width;
            out1[pos] = image[i][j];
        }
	}

    labelFile.read(&x, sizeof(char));
    for (int i = 1; i <= n3; ++i)
		expected[i] = 0.0;
    expected[x + 1] = 1.0;

    return x;
}