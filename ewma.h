#ifndef __EWMA_H__
#define __EWMA_H__

#include "OsqpEigen/OsqpEigen.h"

using namespace std;


// LoopEWMA 父类
class LoopEWMA {
public:

    // 类成员变量
    Eigen::Array<Eigen::MatrixXd, -1, -1> U;
    Eigen::Array<Eigen::MatrixXd, -1, -1> Y;
    Eigen::MatrixXd A0;
    Eigen::MatrixXd B0;
    Eigen::VectorXd target;
    double lamda1;
    bool decay;
    int t;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;
    string control;
    string type;
    int p;
    int m;
    Eigen::MatrixXd At;
    Eigen::MatrixXd Bt;
    Eigen::MatrixXd Ut;


    // 构造函数 + 代码审查
    LoopEWMA(
            Eigen::Array<Eigen::MatrixXd, -1, -1> U_,
            Eigen::Array<Eigen::MatrixXd, -1, -1> Y_,
            Eigen::VectorXd target_,
            double lamda1_,
            Eigen::MatrixXd A0_,
            Eigen::MatrixXd B0_,
            string control_,
            string type_,
            bool decay_,
            int t_,
            Eigen::VectorXd lb_,
            Eigen::VectorXd ub_);


    // 函数：ewma
    Eigen::MatrixXd ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda);

    // 函数：ss_ewma
    Eigen::MatrixXd ss_ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda, int WEIGHT = 15);

    // 函数：tr_ewma
    Eigen::MatrixXd tr_ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda, int t, int T = 15);

    // 函数：tr_ss_ewma
    Eigen::MatrixXd
    tr_ss_ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda, int t, int T = 15, int WEIGHT = 15);

    // 函数：select
    Eigen::MatrixXd select(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda);

    // 函数：intercept_update
    Eigen::MatrixXd intercept_update();

    // 函数：slope_update()
    Eigen::MatrixXd slope_update();

    // 函数：cal_recipe
    Eigen::MatrixXd cal_recipe(Eigen::MatrixXd At, Eigen::MatrixXd Bt);

    // 函数：run
    map <string, Eigen::MatrixXd> run();
};

#endif