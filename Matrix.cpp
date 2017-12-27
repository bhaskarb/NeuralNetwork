#include <random>
#include <iostream>
#include <assert.h>
#include "Matrix.h"

using namespace NN;
//Contructor for the matrix object
Matrix::Matrix(int m)
{
    assert(m > 0);
    row_ = m;
    col_ = m;
    Init_();
}
//Contructor for the matrix object
Matrix::Matrix(int m, int n)
{
    assert(m > 0);
    assert(n > 0);
    row_ = m;
    col_ = n;
    Init_();
}

//Initialize the matrix data structure
void Matrix::Init_(void)
{
    data_ = new double*[row_];
    for(int i = 0; i < row_; i ++) {
        data_[i] = new double[col_];
    }
}

//Initialize the matrix with a random set of values from a normal distribution
void Matrix::randn(double mu, double sigma)
{

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mu, sigma);
    for(int i = 0; i < row_; i ++) {
        for(int j = 0; j < col_; j ++) {
            data_[i][j] = distribution(generator);
        }
    }
}

//Get the value at the location: b = A[i][j]
//The problem with the [] operator is not knowing which dimensions is currently picked
double Matrix::val(int row, int col) const
{
    assert(row < row_);
    assert(row >= 0);
    assert(col < col_);
    assert(col >= 0);
    return data_[row][col];
}

//Get the value at the location: A[i][j] = b
//The problem with the [] operator is not knowing which dimensions is currently picked
void Matrix::set(int row, int col, double value)
{
    assert(row < row_);
    assert(row >= 0);
    assert(col < col_);
    assert(col >= 0);
    data_[row][col] = value;
}

Matrix::~Matrix()
{
    for(int i = 0; i < row_; i ++) {
        delete [] data_[i];
    }
    delete data_;
}

//Overload the + operator to return sum of 2 matrices
Matrix Matrix::operator +(const Matrix &n)
{
    int nrow_, ncol_;

    n.size(&nrow_, &ncol_);
    assert(row_ == nrow_);
    assert(col_ == ncol_);
    Matrix out(row_, col_);
    for (int i = 0; i < row_; i ++) {
        for(int j = 0; j < col_; j ++) {
            out.set(i, j, data_[i][j] + n.val(i, j));
        }
    }
    return out;
}

//Overload the - operator to return sum of 2 matrices
Matrix Matrix::operator -(const Matrix &n)
{
    int nrow_, ncol_;

    n.size(&nrow_, &ncol_);
    assert(row_ == nrow_);
    assert(col_ == ncol_);
    Matrix out(row_, col_);
    for (int i = 0; i < row_; i ++) {
        for(int j = 0; j < col_; j ++) {
            out.set(i, j, data_[i][j] - n.val(i, j));
        }
    }
    return out;
}

Matrix Matrix::operator *(const Matrix &n) 
{
    int nrow_, ncol_;
    double sum;

    n.size(&nrow_, &ncol_);
    assert(nrow_ == col_);
    Matrix out(row_, ncol_);
    for(int i = 0; i < row_; i ++) {
        for(int j = 0; j < ncol_; j ++) {
            sum = 0;
            for(int k = 0; k < col_; k ++) {
                sum += data_[i][k]*n.val(k, j);
            }
            out.set(i, j, sum);
        }
    }
    return out;
}

void Matrix::operator =(const Matrix &n)
{
    int nrow_, ncol_;

    n.size(&nrow_, &ncol_);
    assert(nrow_ == row_);
    assert(ncol_ == col_);
    for(int i = 0; i < row_; i ++) {
        for(int j = 0; j < row_; j ++) {
            data_[i][j] = n.val(i, j);
        }
    }
}
