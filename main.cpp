#include <random>
#include <iostream>
#include "Activation.h"
#include "Matrix.h"
#include "Layer.h"

using namespace NN;

int main()
{
    Layer inlayer(2, 1);
    Matrix in(2, 1);
    logistic logit;

    inlayer.Init(0, 0.1, &logit);
    std::cout << inlayer;
    in.set(0, 0, 1);
    in.set(1, 0, 0);
    std::cout << inlayer.Fprop(in);

}
