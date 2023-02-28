#include<iostream>
#include<string>
#include <Eigen/Dense>
#include "OsqpEigen/OsqpEigen.h"
#include"solver.h"
#include<list>
#include <algorithm>
#include<math.h>
#include<map>

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
    string control; // 后续添加默认值
    string type; // 后续添加默认值
    int p;
    int m;
    Eigen::MatrixXd At;
    Eigen::MatrixXd Bt;
    Eigen::MatrixXd Ut;

    // 构造函数 + 代码审查
    LoopEWMA(
            Eigen::Array<Eigen::MatrixXd, -1, -1> U_,
            Eigen::Array<Eigen::MatrixXd, -1, -1> Y_,
            Eigen::MatrixXd A0_,
            Eigen::MatrixXd B0_,
            Eigen::VectorXd target_,
            double lamda1_,
            bool decay_,
            int t_,
            Eigen::VectorXd lb_,
            Eigen::VectorXd ub_,
            string control_ = "intercept",
            string type_ = "ewma") {


        // U、Y、target审查
        if (U_.size() == 0 || Y_.size() == 0 || target_.size() == 0) {
            throw "U 或 Y 或 target 不能为空！";
        }
        if (target_.size() == 1) {
            cout << "请根据Y(0).size()的值，构造对应维度的target同值向量或矩阵。" << endl;
        }

        // A0、B0、control审查
        transform(control_.begin(), control_.end(), control_.begin(), ::tolower);
        if (control_ != "intercept" && control_ != "i" && control_ != "slope" && control_ != "s") {
            throw "control参数不合法！";
        }
        if (control_ == "i" || control_ == "intercept") {
            if (B0_.size() == 0) {
                throw "当control为'i'或'intercept'时，B0不能为空";
            }
        } else {
            if (A0_.size() == 0) {
                throw "当control为's'或'slope'时，A0不能为空";
            }
        }

        // lb、ub审查
        if (lb_.size() == 1 || ub_.size() == 1) {
            cout << "请根据U(0).size()的值，构造对应维度的lb|ub的同值向量或矩阵。" << endl;
        }


        // 参数初始化
        U = U_;
        Y = Y_;
        A0 = A0_;
        B0 = B0_;
        target = target_;
        lamda1 = lamda1_;
        decay = decay_;
        t = t_;
        lb = lb_;
        ub = ub_;
        control = control_;
        type = type_;
        p = int(Y_(0).rows());
        m = int(U_(0).rows());

    }


    // 函数：ewma
    Eigen::MatrixXd ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda) {
        Eigen::MatrixXd e_val = Eigen::MatrixXd::Zero(int(data_list(0).rows()), int(data_list(0).cols()));
        int len = int(data_list.rows());
        if (len <= 0) {
            throw "data_list数据为空!";
        } else {
            if (len == 1) {
                e_val = 1 * data_list(0);
            } else {
                for (int i = 0; i < len; i++) {
                    if (i > 0) {
                        e_val += lamda * pow(1 - lamda, len - 1 - i) * data_list(i);
                    } else {
                        e_val += pow(1 - lamda, len - 1 - i) * data_list(i);
                    }
                }
            }
        }
        return e_val;

    }


    // 函数：ss_ewma
    Eigen::MatrixXd ss_ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda, int WEIGHT = 15) {
        Eigen::MatrixXd e_val = Eigen::MatrixXd::Zero(int(data_list(0).rows()), int(data_list(0).cols()));
        int len = int(data_list.rows());
        if (len <= 0) {
            throw "data_list数据为空!";
        } else {
            if (len == 1) {
                e_val = 1 * data_list(0);
            } else {
                double a = lamda * (1 + exp(-pow(2, 2) / WEIGHT));
                double b = 1 - lamda * (1 + exp(-pow(2, 2) / WEIGHT));
                map<int, map<int, double>> alpha_mapping;
                alpha_mapping[2][2] = a;
                alpha_mapping[2][1] = b;
                for (int i = 3; i < len + 1; i++) {
                    for (int j = 0; j < i; j++) {
                        if (j == 0) {
                            alpha_mapping[i][i - j] = lamda * (1 + exp(-pow(i, 2) / WEIGHT));
                        } else {
                            alpha_mapping[i][i - j] = (1 - alpha_mapping[i][i]) * alpha_mapping[(i - 1)][(i - j)];
                        }
                    }
                }
                for (int k = 0; k < data_list.rows(); k++) {
                    e_val += alpha_mapping[len][(len - k)] * data_list(len - 1 - k);
                }
            }
        }
        return e_val;
    }


    // 函数：tr_ewma
    Eigen::MatrixXd tr_ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda, int t, int T = 15) {
        double t_lamda = lamda * exp(-t / T);
        return ewma(data_list, t_lamda);
    }


    // 函数：tr_ss_ewma
    Eigen::MatrixXd
    tr_ss_ewma(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda, int t, int T = 15, int WEIGHT = 15) {
        double t_lamda = lamda * exp(-t / T);
        return ss_ewma(data_list, t_lamda, WEIGHT);
    }


    // 函数：select
    Eigen::MatrixXd select(Eigen::Array<Eigen::MatrixXd, -1, -1> data_list, double lamda) {
        Eigen::MatrixXd value;
        if (decay) {
            if (type == "ss-ewma") {
                value = tr_ss_ewma(data_list, lamda, t);
            } else {
                value = tr_ewma(data_list, lamda, t);
            }
        } else {
            if (type == "ss-ewma") {
                value = ss_ewma(data_list, lamda);
            } else {
                value = ewma(data_list, lamda);
            }
        }
        return value;
    }


    // 函数：intercept_update
    Eigen::MatrixXd intercept_update() {
        int row = int(U.rows());
        Eigen::Array<Eigen::MatrixXd, -1, 1> data_list;
        data_list.resize(row, 1);
        for (int i = 0; i < row; i++) {
            data_list(i) = Y(i) - B0 * U(i);
        }
        At = select(data_list, lamda1);
        return At;

    }


    // 函数：slope_update()
    Eigen::MatrixXd slope_update() {
        int row = int(U.rows());
        Eigen::Array<Eigen::MatrixXd, -1, 1> data_list;
        data_list.resize(row, 1);
        for (int i = 0; i < row; i++) {
            Eigen::MatrixXd B_1 = (Y(i) - A0) * U(i).transpose();
            Eigen::MatrixXd B_2 = (U(i) * U(i).transpose() + 0.001 * Eigen::MatrixXd::Identity(m, m)).inverse();
            data_list(i) = B_1 * B_2;
        }
        Bt = select(data_list, lamda1);
        return Bt;

    }


    // 函数：cal_recipe
    Eigen::MatrixXd cal_recipe(Eigen::MatrixXd At, Eigen::MatrixXd Bt) {

        /*计算下一个wafer/lot的配方，考虑到MIMO系统，分为三种情况：
        input个数 = output个数，存在唯一解*/
        if (p == m) {
            if (ub.size() == 0 && lb.size() == 0) {
                // 不考虑约束问题
                Ut = Bt.inverse() * (target - At); // 离线求解
                cout << "p = m,无约束========================================================" << endl;
            } else {
                // 二次规划求解器，有不等式约束
                Eigen::MatrixXd H = Bt.transpose() * Bt;
                Eigen::MatrixXd f = ((At.transpose() * Bt) - (target.transpose() * Bt)).transpose();
                Eigen::MatrixXd L = Eigen::MatrixXd::Identity(m, m);
                Ut = OSQP_solver(H.sparseView(), f, L.sparseView(), lb, ub);
                cout << "p = m,有约束========================================================" << endl;
                cout << "H" << endl << H << endl;
                cout << "f" << endl << f << endl;
                cout << "L" << endl << L << endl;
                cout << "lb" << endl << lb << endl;
                cout << "ub" << endl << ub << endl;
            }
        }
            // output个数 > input个数，无解，使用最小二乘求解最优解
        else if (p > m) {
            if (ub.size() == 0 && lb.size() == 0) {
                Ut = (Bt.transpose() * Bt).inverse() * Bt.transpose() * (target - At);
                cout << "p > m,无约束，最小二乘法========================================================" << endl;
            } else {
                // 二次规划求解器，有不等式约束
                Eigen::MatrixXd H = Bt.transpose() * Bt;
                Eigen::MatrixXd f = ((At.transpose() * Bt) - (target.transpose() * Bt)).transpose();
                Eigen::MatrixXd L = Eigen::MatrixXd::Identity(m, m);
                Ut = OSQP_solver(H.sparseView(), f, L.sparseView(), lb, ub);
                cout << "p > m,有约束========================================================" << endl;
                cout << "H" << endl << H << endl;
                cout << "f" << endl << f << endl;
                cout << "L" << endl << L << endl;
                cout << "lb" << endl << lb << endl;
                cout << "ub" << endl << ub << endl;
            }
        }
            // output个数 < input个数，存在多个解，选择与前一次recipt最接近的解
        else {
            // 利用拉格朗日乘子离线求解
            if (ub.size() == 0 && lb.size() == 0) {
                Eigen::MatrixXd u1 = (Bt * Bt.transpose()).inverse();
                Eigen::MatrixXd u2 = Bt * U(U.rows() - 1) - target + At;
                Ut = U(U.rows() - 1) - (Bt.transpose() * u1 * u2);
                cout << "p < m,无约束，拉格朗日乘子法========================================================" << endl;
            } else {
                // 二次规划求解,小数点后第二位之后，对不上
                Eigen::MatrixXd H = Eigen::MatrixXd::Identity(m, m);
                Eigen::MatrixXd f = -U(U.rows() - 1);
                Eigen::MatrixXd L = Eigen::MatrixXd::Identity(m, m);
                Eigen::MatrixXd k = ub;
                Eigen::MatrixXd Aeq = Bt;
                Eigen::MatrixXd beq = target - At;
                Ut = OSQP_solver(H.sparseView(), f, Aeq.sparseView(), beq, beq);
                cout << "p < m,有约束========================================================" << endl;
                cout << "H" << endl << H << endl;
                cout << "f" << endl << f << endl;
                cout << "L" << endl << L << endl;
                cout << "k" << endl << k << endl;
                cout << "Aeq" << endl << Aeq << endl;
                cout << "beq" << endl << beq << endl;
            }
        }
        return Ut;
    }


    // 函数：run
    map<string, Eigen::MatrixXd> run() {
        map<string, Eigen::MatrixXd> result;
        if (control == "intercept" || control == "i") {
            At = intercept_update();
            Ut = cal_recipe(At, B0);
            cout << "At" << endl << At << endl << "Ut" << endl << Ut << endl;
            Eigen::MatrixX2d None;
            result["recipe"] = Ut;
            result["At"] = At;
            result["Bt"] = None;
        } else {
            Bt = slope_update();
            Ut = cal_recipe(A0, Bt);
            cout << "Bt" << endl << Bt << endl << "Ut" << endl << Ut << endl;
            Eigen::MatrixX2d None;
            result["recipe"] = Ut;
            result["At"] = None;
            result["Bt"] = Bt;
        }
        return result;

    }


};


int main() {

    // 定义None值
    Eigen::Array<Eigen::MatrixXd, 0, 0> V;
    //cout << "向量V的None值：" << V.size() << endl; // 打印结果为0


    // 配置U,Y矩阵数组
    Eigen::Array<Eigen::MatrixXd, 3, 1> U; // 前n次的输入
    Eigen::Array<Eigen::MatrixXd, 3, 1> Y; // 前n次的量测值
    U(0) = Eigen::MatrixXd{{0.0},
                           {0.0}};
    U(1) = Eigen::MatrixXd{{-0.622},
                           {-1.421}};
    U(2) = Eigen::MatrixXd{{-1.099},
                           {-2.198}};
    Y(0) = Eigen::MatrixXd{{1.831},
                           {3.216}};
    Y(1) = Eigen::MatrixXd{{1.047},
                           {1.453}};
    Y(2) = Eigen::MatrixXd{{0.714},
                           {-1.381}};


    // 配置A0,B0矩阵
    Eigen::MatrixXd A0{ // 初始截距，有DOE和回归技术估计获得
            {1.0},
            {0.3}
    };
    Eigen::MatrixXd B0{ // 初始斜率，有DOE和回归技术估计获得
            {1.0, 0.2},
            {0.3, 1.0}
    };


    // 配置目标矩阵
    Eigen::VectorXd target(2);  // 期望目标值
    target << 0, 0;


    // 配置其他参数
    double lamda1 = 0.2;  // 当前截距/斜率的权重因子
    bool is_decay = true;  // 衰减权重因子


    // 配置运行次数
    int t = 0;


    // 配置约束
    Eigen::VectorXd lb(2); // 输入的下界
    Eigen::VectorXd ub(2); // 输入的上界
    lb << -100, -100;
    ub << 100, 100;


    // 配置更新方式和ewma类型
    string control = "i";  // 可选截距更新或者斜率更新
    string type = "ewma";  // 可选ewma和ss-ewma


    // 运行
    LoopEWMA loopewma(U, Y, A0, B0, target, lamda1, is_decay, t, lb, ub, control, type); //必须全部输入
    map<string, Eigen::MatrixXd> result = loopewma.run();
    cout << "********************************* result ***************************************" << endl;
    cout << "recipe：" << endl << result["recipe"] << endl;
    cout << "At：" << endl << result["At"] << endl;
    cout << "Bt：" << endl << result["Bt"] << endl;
};