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

