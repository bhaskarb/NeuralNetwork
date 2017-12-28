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
    W_ = new Matrix(ninputs_ + 1, noutputs_);
    dEdW_ = new Matrix(ninputs_ + 1, noutputs_);
    Z_ = new Matrix(noutputs_, 1);
    X_ = new Matrix(ninputs_ + 1, 1);
    dYdZ_ = new Matrix(noutputs_, 1);

    //Initialize the weights, right now using a normal distribution
    W_->randn(mu, sigma);
}

Layer::~Layer()
{
    delete W_;
    delete dEdW_;
    delete dYdZ_;
    delete Z_;
    delete X_;
}

/*-----------------------------------------------------------------------------------
 * Do the forward propagation algorithm
 * Take the input matrix x(ninputs_x1) and return the layers forward output
 * W'x + b, the way this is implemented if that b is assumed to be a part of the weights
 * with x being 1 for that
 -----------------------------------------------------------------------------------*/
Matrix Layer::Fprop(const Matrix &x)
{
    Matrix out(noutputs_, 1);
    Matrix localX(ninputs_ + 1 , 1);
    localX = x.resize(ninputs_ + 1, 1, 1.0);
    *X_ = localX; 
    *Z_ = W_->transpose()*(*X_);
    out = f_(*Z_);
    return out;
}

/*-----------------------------------------------------------------------------------
 * Backpropagation algorithm
 * Given the derivatives at the output node, backprop gives a way to propagate the
 * derivatives back while updateing the layers interal weights
 * dEdX = dZdX*(dYdZ o dEdZ) is the input derivatives given the output
 * dEdW = dZdW*(dYdZ o dEdZ) weight updates due to propagation
 -----------------------------------------------------------------------------------*/
Matrix Layer::Bprop(const Matrix &dEdY)
{
    Matrix dEdX(ninputs_ + 1, 1);

    //dEdY = outx1
    //dYdZ = outx1
    //dZdX = (in + 1)xout=W_
    //dEdX = (in + 1)x1 = dZdX*(dYdZ o dEdZ);
    dEdX = (*W_)*(dEdY && (*dYdZ_));
    //dZdW = (in + 1)*1 = X_, need to prove this
    //dEdW = (in +1)xout = dZdW*(dYdZ o dEdZ);
    *dEdW_ = (*X_)*(dEdY&&(*dYdZ_));
    return dEdX;
}

//Calulate the activation of the array
Matrix Layer::f_(const Matrix &m)
{
    int row, col;

    m.size(&row, &col);
    Matrix out(row, col);
    for(int i = 0; i < row; i ++) {
        for(int j = 0; i < col; j ++) {
            out.set(i, j, output(m.val(i,j)));
        }
    }
    return out;
}

//Calulate the activation derivative of the array
Matrix Layer::fprime_(const Matrix &m)
{
    int row, col;

    m.size(&row, &col);
    Matrix out(row, col);
    for(int i = 0; i < row; i ++) {
        for(int j = 0; i < col; j ++) {
            out.set(i, j, outputprime(m.val(i,j)));
        }
    }
    return out;
}
