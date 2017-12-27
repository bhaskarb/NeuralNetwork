#ifndef __LAYER_H__
#define __LAYER_H__
#include "Matrix.h"
#include "Activation.h"

namespace NN {

    class Layer {
        public:
            Layer(int num_inputs, int num_outputs);
            ~Layer();
        private:
            int ninputs_, noutputs_;
            Matrix weights;
            Matrix bias;
    }
}
#endif
