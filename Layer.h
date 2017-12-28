#ifndef __LAYER_H__
#define __LAYER_H__
#include "Matrix.h"
#include "Activation.h"

namespace NN {
    class Layer {
        public:
            Layer(int num_inputs, int num_outputs);
            ~Layer();
            //Initialize the layer structures, creates the weight, bias and the derivative matrices
            void Init(double mu = 0.0, double sigma = 1.0);
            Matrix Fprop(const Matrix &x);
            Matrix Bprop(const Matrix &dEdY);

            Matrix& W(void) const {return *W_; };
            Matrix& dEdW(void) const {return *dEdW_; };
            const int noutputs(void) const {return ninputs_;};
            const int ninputs(void) const {return ninputs_;};

            //print the matrix
            friend std::ostream &operator<<(std::ostream &os, const Layer &l) 
            {
                std::cout << "Layer inputs = " << l.ninputs() << '\n';
                std::cout << "Layer outputs = " << l.noutputs() << '\n';
                std::cout << "Layer weights = " << l.W() << '\n';
            }

        private:
            //We are making an ABS here, the various layer types will define this
            virtual double output(double x) = 0;
            virtual double outputprime(double x) = 0;

            Matrix f_(const Matrix &m);
            Matrix fprime_(const Matrix &m);
            int ninputs_, noutputs_;

            //We are incorporating the bias into the weight vector so the vector is (ninputs_ + 1)*noutputs_
            Matrix *W_, *dEdW_;
            //Keep track of the fprop inputs and outputs and the derivatives since this is needed by backprop
            Matrix *dYdZ_;
            Matrix *Z_;
            Matrix *X_;
            Activation *act;
    };
}
#endif
