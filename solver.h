#ifndef __SOLVER_H__
#define __SOLVER_H__

#include <iostream>
#include <Eigen/Dense>
#include "OsqpEigen/OsqpEigen.h"


// 1. 声明QPST求解器
Eigen::VectorXd OSQP_solver( //返回的是向量
        Eigen::SparseMatrix<double, Eigen::ColMajor, int> P, Eigen::VectorXd q,
        Eigen::SparseMatrix<double, Eigen::ColMajor, int> G, Eigen::VectorXd lb, Eigen::VectorXd ub);


#endif