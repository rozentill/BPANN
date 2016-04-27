#include<iostream>
#include<cmath>
#include<stdlib.h>
#include<time.h>

#define random(x) (rand()%x)
using namespace std;

class BPANN{
private:

    /* s1, s2, s3 */
    int numInput;
    int numOutput;
    int numHidden;

    /* w1 and w2 include b1 and b2 */
    float ** weightInputHidden;
    float ** weightHiddenOutput;

    /* a1 and a2 */
    float * outputHidden;
    float * outputOutput;

    /* sensitivity */
    float * sOutput;
    float * sHidden;

    /* e */
    float errorTrain;
    float errorTest;

    /* activation function */
    float (*activation)(float);
    float (*dactivation)(float);

    float dtanh(float x){
        return 1-pow(x,2);
    }

    float sigmoid(float x){
        return 1/(1+exp(-x));
    }

    float dsigmoid(float x){
        return (1-1/(1+exp(-x)))*(1/(1+exp(-x)));
    }

public:

    BPANN(int ni,int no,int nh, int funcNo = 0){

        /* number of nodes in three */
        numInput = ni;
        numOutput = no;
        numHidden = nh;

        /* weight matrix */
        weightInputHidden = new float * [numInput+1];//input-hidden and bias node
        weightHiddenOutput = new float * [numHidden+1];//hidden-output and bias node

        srand((int)time(0));

        /* generate matrix with random values */
        for (int i=0; i<=numInput; i++) {
            weightInputHidden[i] = new float[numHidden];
            for (int j = 0; j < numHidden; j++) {
              weightInputHidden[i][j] = random(4)/10.0-0.2;
            }
        }
        for (int i=0; i<=numHidden; i++) {
            weightHiddenOutput[i] = new float[numOutput];
            for (int j = 0; j < numOutput; j++) {
              weightHiddenOutput[i][j] = random(4)-2;
            }
        }

        /* initialize output matrix of each layer a0, a1 and a2 */
        outputInput = new float[numInput];
        outputHidden = new float[numHidden];
        outputOutput = new float[numOutput];

        /* initialize error */
        errorTrain = 0;
        errorTest = 0;

        /* activation functions choices */
        switch (funcNo) {
            case 0:
                activation = sigmoid;
                dactivation = dsigmoid;
                break;
            case 1:
                activation = tanh;
                dactivation = dtanh;
                break;
        }

    }

    void feedForward(float * input){


        /* compute a0 */
        cout << "Computing a0..." << "\n";
        for (int i = 0; i < numInput; i++) {
            outputInput[i] = input[i];
        }

        /* compute a1 */
        cout << "Computing a1..." << "\n";
        for (int i = 0; i < numHidden; i++) {
            double sum = 0;
            for (int j = 0; j < numInput; j++) {
                sum += weightInputHidden[j][i]*outputInput[j]+weightInputHidden[numInput][j];
            }
            outputHidden[i] = activation(sum);
        }

        /* compute a2 */
        cout << "Computing a2..." << "\n";
        for (int i = 0; i < numOutput; i++) {
            double sum = 0;
            for (int j = 0; j < numHidden; j++) {
                sum += weightHiddenOutput[j][i]*outputHidden[j]+weightHiddenOutput[numOutput][j];
            }
            outputOutput[i] = activation(sum);
        }
    }

    float backPropagate(float * target){
        float error;
        /* begin with s2 */
        cout << "Computing s2..." << "\n";
        for (int i = 0; i < numOutput; i++) {
            sOutput[i] = -2 * (target[i]-outputOutput[i]);
            error += (target[i]-outputOutput[i])*(target[i]-outputOutput[i]);
        }

        /* move on to s1 */
        cout << "Computing s1..." << "\n";
        for (int i = 0; i < numHidden; i++) {
            float tmp = 0;
            for (int j = 0; j < numOutput; j++) {
                tmp += weightHiddenOutput[i][j]*sOutput[j];
            }
            sHidden[i] = dactivation(outputHidden[i]) * tmp;
        }

        return error;
    }

    void updateWeight(float rate){
        cout << "Updating w2..." << "\n";
        /* update weight w2 */
        for (int i = 0; i < numHidden ; i++) {
            for (int j = 0; j < numOutput; j++) {
                weightHiddenOutput[i][j] = weightHiddenOutput[i][j] - rate * outputHidden[i] * sOutput[j];
            }
        }

        /* update weight w1 */
        cout << "Updating w1..." << "\n";
        for (int i = 0; i < numInput; i++) {
            for (int j = 0; j < numHidden; j++) {
                weightInputHidden[i][j] = weightInputHidden[i][j] - rate * outputInput[i] * sHidden[j];
            }
        }

        /* update bias b2 */
        cout << "Updating b2..." << "\n";
        for (int i = 0; i < numOutput; i++) {
            weightHiddenOutput[numHidden][i] = weightHiddenOutput[numHidden][i] - rate * sOutput[i];
        }

        /* update bias b1 */
        cout << "Updating b1..." << "\n";
        for (int i = 0; i < numHidden; i++) {
            weightInputHidden[numInput][i] = weightInputHidden[numInput][i] - rate * sHidden[i];
        }
    }

    void train(float *** pattern, float rate = 0.2, iterations = 1000){
        int numData = len(pattern);
        for (int i = 0; i < iterations; i++) {
            errorTrain = 0;
            for (int j = 0; j < numData; j++) {
                /* input = pattern[j][0] , target = pattern[j][1] */
                feedForward(pattern[j][0]);
                errorTrain += backPropagate(pattern[j][1]);
                updateWeight(rate);
            }
            errorTrain /= 2;
            cout << "This training iteration's loss is :" << errorTrain << ".\n";
        }
    }

    void test(float *** pattern){
      int numData = len(pattern);
      int hitNum = 0;
      bool ifHit = TRUE;
      errorTest = 0;
        for (int j = 0; j < numData; j++) {
            /* input = pattern[j][0] , target = pattern[j][1] */
            feedForward(pattern[j][0]);
            errorTest += backPropagate(pattern[j][1]);
            ifHit = TRUE;
            for (int i = 0; i < numOutput; i++) {
                if ((outputOutput[i]-pattern[j][1][i]>1)||(outputOutput[i]-pattern[j][1][i]<-1)) {
                    ifHit = FALSE;
                    break;
                }
            }
            if (ifHit) {
                hitNum ++;
            }
        }
        errorTest /= 2;
        cout << "The test's loss is :" << errorTest << ".\n";
        cout << "The hit rate is :"<< hitNum/(float)numData << ".\n";
    }

    void saveModel(char * filename){

    }

    void loadModel(char * filename){

    }

};
