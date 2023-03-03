#ifndef __REWMA_H__
#define __REWMA_H__

#include "OsqpEigen/OsqpEigen.h"


using namespace std;


// 1. RecursionEWMA类
class RecursionEWMA {
public:

    // 类成员变量
    Eigen::MatrixXd Ut_1;
    Eigen::MatrixXd Yt;
    Eigen::MatrixXd A0;
    Eigen::MatrixXd B0;
    Eigen::MatrixXd At_1;
    Eigen::MatrixXd Bt_1;
    Eigen::MatrixXd At;
    Eigen::MatrixXd Bt;
    Eigen::MatrixXd Ut;
    Eigen::VectorXd target;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;
    int t;
    int p;
    int m;
    int T;
    double lamda1;
    bool decay;
    string type;
    string control;


    // 构造函数 + 代码审查
    RecursionEWMA(
            Eigen::MatrixXd Ut_1_,
            Eigen::MatrixXd Yt_,
            Eigen::VectorXd target_,
            double lamda1_,
            Eigen::MatrixXd A0_,
            Eigen::MatrixXd B0_,
            string control_,
            bool decay_,
            int t_,
            Eigen::VectorXd lb_,
            Eigen::VectorXd ub_,
            Eigen::MatrixXd At_1_,
            Eigen::MatrixXd Bt_1_
    );

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