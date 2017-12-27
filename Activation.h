#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__
#include <math.h>

namespace NN {
    class Activation {
        virtual double val(double x) = 0;
        virtual double prime(double x) = 0;
    };
    class logistic: public Activation {
        double val(double x) { return 1.0/(1.0 + exp(-x)); }
        double prime(double x) { return val(x)*(1.0 - val(x)); }
    };
    class Tanh: public Activation {
        double val(double x) { return tanh(x); }
        double prime(double x) { return 1.0 - val(x)*val(x); }
    };
    class relu: public Activation {
        double val(double x) { if(x >= 0) return x; else return 0; }
        double prime(double x) { if(x >= 0) return 1; else return 0; }
    };
}

#endif
