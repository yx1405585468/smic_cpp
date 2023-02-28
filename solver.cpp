//三元组代码
#include <iostream>
#include <Eigen/Dense>
#include "OsqpEigen/OsqpEigen.h"


// 1. 定义、实现QPST求解器
Eigen::VectorXd OSQP_solver( //返回的是向量
        Eigen::SparseMatrix<double, Eigen::ColMajor, int> P, Eigen::VectorXd q,
        Eigen::SparseMatrix<double, Eigen::ColMajor, int> G, Eigen::VectorXd lb, Eigen::VectorXd ub) {


    //定义异常向量
    Eigen::Vector2d Error(2);
    Error << 1, 1;

    // 实例化求解器
    OsqpEigen::Solver solver;

    // 设置
    //solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    // 设置qp求解器的初始数据
    // 矩阵G为m*n矩阵
    solver.data()->setNumberOfVariables(int(G.cols())); //设置G矩阵的列数，即n
    solver.data()->setNumberOfConstraints(int(G.rows())); //设置G矩阵的行数，即m
    if (!solver.data()->setHessianMatrix(P)) return Error;//设置P矩阵
    if (!solver.data()->setGradient(q)) return Error; //设置q or f矩阵。当没有时设置为全0向量
    if (!solver.data()->setLinearConstraintsMatrix(G)) return Error;//设置线性约束的A矩阵
    if (!solver.data()->setLowerBound(lb)) return Error;//设置下边界
    if (!solver.data()->setUpperBound(ub)) return Error;//设置上边界

    // 实例化求解器
    if (!solver.initSolver()) return Error;

    // 求解qp问题
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return Error;

    // 输出控制器输入
    return solver.getSolution();
}


// 2. 制造测试数据，测试求解器函数
int test() {


    //设置P矩阵
    std::vector<Eigen::Triplet < double>>
    HList = { Eigen::Triplet<double>(0, 0, 1),
              Eigen::Triplet<double>(0, 1, -1),
              Eigen::Triplet<double>(0, 2, 1),
              Eigen::Triplet<double>(1, 0, -1),
              Eigen::Triplet<double>(1, 1, 2),
              Eigen::Triplet<double>(1, 2, -2),
              Eigen::Triplet<double>(2, 0, 1),
              Eigen::Triplet<double>(2, 1, -2),
              Eigen::Triplet<double>(2, 2, 4) };
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> P(3, 3);
    P.setFromTriplets(HList.begin(), HList.end());
    std::cout << "矩阵P:" << std::endl << P << std::endl;

    //设置q向量
    Eigen::VectorXd q(3);
    q << 2, -3, 1;
    std::cout << "向量q:" << std::endl << q << std::endl;

    //设置G矩阵
    std::vector<Eigen::Triplet < double>>
    CList = {
            Eigen::Triplet<double>(0, 0, 1),
            Eigen::Triplet<double>(0, 1, 0),
            Eigen::Triplet<double>(0, 2, 0),
            Eigen::Triplet<double>(1, 0, 0),
            Eigen::Triplet<double>(1, 1, 1),
            Eigen::Triplet<double>(1, 2, 0),
            Eigen::Triplet<double>(2, 0, 0),
            Eigen::Triplet<double>(2, 1, 0),
            Eigen::Triplet<double>(2, 2, 1),
            Eigen::Triplet<double>(3, 0, 1),
            Eigen::Triplet<double>(3, 1, 1),
            Eigen::Triplet<double>(3, 2, 1) };
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> G(4, 3);
    G.setFromTriplets(CList.begin(), CList.end());
    std::cout << "不等约束的矩阵G：\n" << G << std::endl;

    //设置约束lb
    Eigen::VectorXd lb(4); //l
    lb << 0, 0, 0, 0.5;
    std::cout << "下界约束lb:" << std::endl << lb << std::endl;

    //设置约束ub
    Eigen::VectorXd ub(4); //u
    ub << 1, 1, 1, 0.5;
    std::cout << "上界约束ub:" << std::endl << ub << std::endl;

    // 导入求解器，求解结果
    Eigen::VectorXd QPSolution;
    QPSolution = OSQP_solver(P, q, G, lb, ub);
    std::cout << "QPSolution:" << std::endl << QPSolution << std::endl;
    return 0;
}