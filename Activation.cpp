#include "Activation.h"

using namespace NN;
//Calulate the activation or derivative of activation for the matrix 
Matrix Activation::f(const Matrix &m, bool derivative)
{
    int row, col;

    m.size(&row, &col);
    Matrix out(row, col);
    for(int i = 0; i < row; i ++) {
        for(int j = 0; j < col; j ++) {
            if(derivative) {
                out.set(i, j, prime(m.val(i,j)));
            } else {
                out.set(i, j, val(m.val(i,j)));
            }
        }
    }
    return out;
}

