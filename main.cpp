#include <random>
#include <iostream>
#include "Activation.h"
#include "Matrix.h"
#include "Layer.h"
#include "debug.h"
using namespace NN;

double trainXorOnce(Layer& inLayer, Layer& outLayer, Matrix& in, Matrix &t)
{
    Matrix out(1, 1), temp(3, 1), error(1, 1), dE(1,1);

    temp = inLayer.Fprop(in);
    DEBUG_MSG("Propagating from input layer \n");
    out = outLayer.Fprop(temp);
    DEBUG_MSG("Propagating to output layer:\n");
    dE = out - t;
    DEBUG_MSG("Calculating Error \n");
    error = (dE*dE);
    std::cout << "Error: "<< 0.5*error.val(0, 0) << " expected:"<< t.val(0,0) << ", got:" << out.val(0,0) << "\n";
    DEBUG_MSG("Propagating back from output layer \n");
    temp = outLayer.Bprop(dE);
    DEBUG_MSG("Propagating back from hidden layer \n");
    inLayer.Bprop(temp);
    return error.val(0,0);
}

int main()
{
    Layer inLayer(2, 3);
    Layer outLayer(3, 1);
    Matrix in(2, 1), temp(3, 1), out(1, 1), t(1, 1), inLayerdW(3, 3), outLayerdW(4, 1);
    logistic logit;
    double error;
    int steps = 0;

    inLayer.Init(0, 0.1, &logit);
    outLayer.Init(0, 0.1, &logit);
   
    do { 
        inLayerdW.reset();
        outLayerdW.reset();
        error = 0;
        steps ++;
        for(int i = 0; i < 2; i ++) {
            for(int j = 0; j < 2; j ++) {
                DEBUG_MSG(inLayer.W());
                DEBUG_MSG(outLayer.W());
                in.set(0, 0, i);
                in.set(1, 0, j);
                t.set(0, 0, (i==j)?0:1);
                error += trainXorOnce(inLayer, outLayer, in, t);
                inLayerdW += inLayer.dEdW();
                outLayerdW += outLayer.dEdW();
            }
        }
        inLayer.updateW(inLayerdW*4);
        outLayer.updateW(outLayerdW*4);
    } while(error > 4e-5);
    std::cout << inLayer.W();
    std::cout << outLayer.W();
    std::cout << "Steps:" << steps << ", error: " << error << '\n';
}
