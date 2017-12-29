#include <random>
#include <iostream>
#include "Activation.h"
#include "Matrix.h"
#include "Layer.h"
#include "debug.h"
using namespace NN;

void trainXorOnce(Layer& inLayer, Layer& outLayer, Matrix& in, Matrix &t)
{
    Matrix out(1, 1), temp(3, 1), error(1, 1);

    temp = inLayer.Fprop(in);
    DEBUG_MSG("Propagating from input layer \n");
    out = outLayer.Fprop(temp);
    DEBUG_MSG("Propagating to output layer:\n");
    out = out - t;
    DEBUG_MSG("Calculating Error \n");
    error = (out*out);
    std::cout << "Error: "<< error.val(0, 0) << " expected:"<< t.val(0,0) << ", got:" << out.val(0,0) << "\n";
    DEBUG_MSG("Propagating back from output layer \n");
    temp = outLayer.Bprop(out);
    DEBUG_MSG("Propagating back from hidden layer \n");
    inLayer.Bprop(temp);
}

int main()
{
    Layer inLayer(2, 3);
    Layer outLayer(3, 1);
    Matrix in(2, 1), temp(3, 1), out(1, 1), t(1, 1);
    logistic logit;

    inLayer.Init(0, 0.1, &logit);
    outLayer.Init(0, 0.1, &logit);
   
    for(int k = 0; k < 10; k ++) { 
        for(int i = 0; i < 2; i ++) {
            for(int j = 0; j < 2; j ++) {
                in.set(0, 0, i);
                in.set(1, 0, j);
                t.set(0, 0, (i==j)?0:1);
                trainXorOnce(inLayer, outLayer, in, t);
                inLayer.updateW(inLayer.dEdW());
                outLayer.updateW(outLayer.dEdW());
            }
        }
    }
    std::cout << inLayer.W();
    std::cout << outLayer.W();
}
