#include "lib.h"

double diff1[n1 + 1][n2 + 1], diff2[n2 + 1][n3 + 1], error2[n2 + 1], error3[n3 + 1];

const int epochs = 5; //512
const double learningRate = 10 * 1e-3; // 1e-3
const double momentum = 0.9;
const double epsilon = 1e-3;

void initNetwork() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            w1[i][j] = 1.0 * (rand() % 6) / 10;
            if (rand() % 2)
				w1[i][j] = - w1[i][j];
        }
	}
	
	for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            w2[i][j] = 1.0 * (rand() % 10 + 1) / (10 * n3);
            if (rand() % 2)
				w2[i][j] = - w2[i][j];
        }
	}
}

void backPropagation() {
    double sum;

    for (int i = 1; i <= n3; ++i)
        error3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);

    for (int i = 1; i <= n2; ++i) {
        sum = 0;
        for (int j = 1; j <= n3; ++j)
            sum += w2[i][j] * error3[j];

        error2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            diff2[i][j] = (learningRate * error3[j] * out2[i]) + (momentum * diff2[i][j]);
            w2[i][j] += diff2[i][j];
        }
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1 ; j <= n2 ; j++ ) {
            diff1[i][j] = (learningRate * error2[j] * out1[i]) + (momentum * diff1[i][j]);
            w1[i][j] += diff1[i][j];
        }
	}
}

int learningProcess() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j)
			diff1[i][j] = 0;
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j)
			diff2[i][j] = 0;
	}

    for (int i = 1; i <= epochs; ++i) {
        feedForward();
        backPropagation();
        if (resultError() < epsilon)
			return i;
    }
    return epochs;
}

void saveNetwork(string saveFile) {
    ofstream save(saveFile.c_str(), ios::out);

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j)
			save << w1[i][j] << " ";
		save << "\n";
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j)
			save << w2[i][j] << " ";
        save << "\n";
    }
	save.close();
}

int main() {
    imageFile.open(trainingImageFile.c_str(), ios::in | ios::binary);
    labelFile.open(trainingLabelFile.c_str(), ios::in | ios::binary );

    char x;
    for (int i = 1; i <= 16; ++i)
        imageFile.read(&x, sizeof(char));
    for (int i = 1; i <= 8; ++i)
        labelFile.read(&x, sizeof(char));

    initNetwork();
    clock_t startTime = clock();
    
    for (int i = 1; i <= nrTrain; ++i) {
        if (i % 300 == 0) {
            double seconds = 1.0 * (clock() - startTime) / CLOCKS_PER_SEC;
            cout << "Nr " << i << " " << fixed << setprecision(3) << seconds << "\n";

            saveNetwork(networkFile);
        }

        readImage();
		
        int nIterations = learningProcess();
    }
    saveNetwork(networkFile);

    imageFile.close();
    labelFile.close();

    return 0;
}
