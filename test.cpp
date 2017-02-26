#include "lib.h"

void loadNetwork(string file_name) {
	ifstream file(file_name.c_str(), ios::in);

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j)
			file >> w1[i][j];
    }
	
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j)
			file >> w2[i][j];
    }
	
	file.close();
}

int main() {
    imageFile.open(testingImageFile.c_str(), ios::in | ios::binary);
    labelFile.open(testingLabelFile.c_str(), ios::in | ios::binary );

    char x;
    for (int i = 1; i <= 16; ++i)
        imageFile.read(&x, sizeof(char));
    for (int i = 1; i <= 8; ++i)
        labelFile.read(&x, sizeof(char));

    loadNetwork(networkFile);
    
    int nrCorrect = 0;
    for (int i = 1; i <= nrTests; ++i) {
        if (i % 100 == 0) {
            double percent = 1.0 * nrCorrect / i * 100.0;
            cout << "Correct images: " << nrCorrect << " / " << i << " Percent : " << fixed << setprecision(2) << percent << "\n";
        }

        int labelFile = readImage();
		
		feedForward();
        
        int predict = 1;
        for (int i = 2; i <= n3; ++i) {
			if (out3[i] > out3[predict]) {
				predict = i;
			}
		}
		--predict;
		
		if (labelFile == predict)
			++nrCorrect;
    }

    double percent = 1.0 * nrCorrect / nrTests * 100.0;
    cout << "Correct images: " << nrCorrect << " / " << nrTests << " Percent : " << fixed << setprecision(2) << percent << "\n";

    imageFile.close();
    labelFile.close();
    
    return 0;
}
