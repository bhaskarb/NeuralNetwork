#ifndef __MATRIX_H__
#define __MATRIX_H__
#include <iostream>
#include <vector>

namespace NN {
    class Matrix {
        public:
            //constructor
            Matrix();
            Matrix(int m, int n);
            Matrix(int m); //Square matrix
            //Destructor
            ~Matrix(); 

            //Other ways of initializing
            void randn(double mu, double sigma);
            void eye();
            
            void size(int *row, int *col) const { *row = row_; *col = col_; }
            double val(int row, int col) const;
            void set(int row, int col, double value);
          
            //Transpose
            Matrix transpose(void) const;
            Matrix resize(int row, int col, double fill) const;
            //Assignment operator
            void operator =(const Matrix &n);
            //Addition of a matrix to another +=, -= operators. This leads to less memory?
            Matrix& operator +=(const Matrix &n);
            //Unary operator
            Matrix operator -(void) const;

            Matrix operator +(const Matrix &n) const;
            Matrix operator -(const Matrix &n) const;
            //product of 2 matrices
            Matrix operator *(const Matrix &n) const;
            // Y = kA
            Matrix operator *(const int &k) const;
            //Hadamard product
            Matrix operator &&(const Matrix &n) const;
            //equality
            bool operator ==(const Matrix &n) const;
            
            //print the matrix
            friend std::ostream &operator<<(std::ostream &os, const Matrix &m) 
            {
                int mrow_, mcol_;

                m.size(&mrow_, &mcol_);
                os << '\n';
                for(int i = 0; i < mrow_; i ++) {
                    for(int j = 0; j < mcol_; j ++) {
                        os << m.val(i, j) << " ";
                    }
                    os << '\n';
                }
                return os;
            }
        
        private:
            //Right now for simplicity lets just deal with double arrays 
            //since that is what we need for neural computation
            double **data_;
            // row_ -> number of rows
            // col_ -> number of columns
            int row_, col_;
            void Init_(void);
    };
}

#endif
