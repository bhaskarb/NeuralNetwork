#include <cassert> 
#include "Layer.h"
using namespace NN;

Layer::Layer(int num_inputs, int num_outputs)
{
    ninputs_= num_inputs;
    noutputs_ = num_outputs;
}

//Initialize the weights using a provided mu and sigma
void Layer::Init(double mu, double sigma)
{
    assert(ninputs_ > 0);
    assert(noutputs_ > 0);
    weight_ = new Matrix(ninputs_, noutputs_);
    weightderiv_ = new Matrix(ninputs_, noutputs_);
    bias_ = new Matrix(noutputs_, 1);
    biasderiv_ = new Matrix(noutputs_, 1);

    weight_->randn(mu, sigma);
    bias_->randn(mu, sigma);
}

Layer::~Layer()
{
    delete weight_;
    delete weightderiv_;
    delete bias_;
    delete biasderiv_;
}

//Calulate the activation of the array
Matrix& Layer::f_(const Matrix &m)
{
    int row, col;

    m.size(row, col);
    Matrix out(row, col);
    for(int i = 0; i < row; i ++) {
        for(int j = 0; i < col; j ++) {
            out.set(i, j, output(m.val(i,j)));
        }
    }
    return out;
}

//Calulate the activation derivative of the array
Matrix& Layer::fprime_(const Matrix &m)
{
    int row, col;

    m.size(row, col);
    Matrix out(row, col);
    for(int i = 0; i < row; i ++) {
        for(int j = 0; i < col; j ++) {
            out.set(i, j, outputprime(m.val(i,j)));
        }
    }
    return out;
}
