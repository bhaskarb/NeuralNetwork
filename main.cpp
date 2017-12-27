#include <random>
#include <iostream>
#include "Matrix.h"
#include "Layer.h"

using namespace NN;

int main()
{
    Matrix m(100000, 10000);
    Matrix n(10000, 50000);

    m.randn(0, 10);
    n.randn(0, 1);
}
