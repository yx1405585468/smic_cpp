#include"r_ewma.h"
#include"solver.h"

using namespace std;


// 1. RecursionDEWMA子类
class RecursionDEWMA : public RecursionEWMA {
public:
    // 类成员变量


    Eigen::MatrixXd Dt_1;
    Eigen::MatrixXd D0;
    double lamda2 = 0.3;
    Eigen::MatrixXd Dt;


    // 构造函数
    RecursionDEWMA(
            Eigen::MatrixXd Ut_1_,
            Eigen::MatrixXd Yt_,
            Eigen::MatrixXd At_1_,
            Eigen::MatrixXd Bt_1_,
            Eigen::MatrixXd Dt_1_,
            Eigen::MatrixXd A0_,
            Eigen::MatrixXd B0_,
            Eigen::MatrixXd D0_,
            Eigen::VectorXd target_,
            Eigen::VectorXd lb_,
            Eigen::VectorXd ub_,
            string control_,
            bool decay_,
            int t_,
            double lamda1_,
            double lamda2_) : RecursionEWMA(Ut_1_, Yt_, target_, lamda1_, A0_, B0_, control_, decay_, t_, lb_, ub_,
                                            At_1_, Bt_1_), Dt_1(Dt_1_), D0(D0_), lamda2(lamda2_) {
    }

    // 函数：drift_update
    Eigen::MatrixXd drift_update() {

        if (control == "intercept" || control == "i") {
            Dt = lamda2 * (Yt - B0 * Ut_1 - At_1) + (1 - lamda2) * Dt_1;
        } else {
            Eigen::MatrixXd replace = Eigen::MatrixXd::Ones(Yt.rows(), Yt.cols()) * (Bt_1 * Ut_1)(0, 0);
            Dt = lamda2 * (Yt - A0 - replace) + (1 - lamda2) * Dt_1;
        }
        return Dt;
    }

    // 函数：intercept_update
    Eigen::MatrixXd intercept_update() {
        double t_lamda;
        if (decay) {
            t_lamda = lamda1 * exp(-t / T);
        } else {
            t_lamda = lamda1;
        }
        Eigen::MatrixXd At = t_lamda * (Yt - B0 * Ut_1) + (1 - t_lamda) * At_1;
        return At;

    }

    // 函数：cal_recipe
    Eigen::MatrixXd cal_recipe(Eigen::MatrixXd At, Eigen::MatrixXd Bt, Eigen::MatrixXd Dt) {

        /*计算下一个wafer/lot的配方，考虑到MIMO系统，分为三种情况：
        input个数 = output个数，存在唯一解*/
        if (p == m) {
            if (ub.size() == 0 && lb.size() == 0) {
                // 不考虑约束问题
                Ut = Bt.inverse() * (target - At - Dt); // 离线求解
                cout << "p = m,无约束========================================================" << endl;
            } else {
                // 二次规划求解器，有不等式约束
                Eigen::MatrixXd H = Bt.transpose() * Bt;
                Eigen::MatrixXd f = ((At.transpose() * Bt) + (Dt.transpose() * Bt) -
                                     (target.transpose() * Bt)).transpose();
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
                Ut = (Bt.transpose() * Bt).inverse() * Bt.transpose() * (target - At - Dt);
                cout << "p > m,无约束，最小二乘法========================================================" << endl;
            } else {
                // 二次规划求解器，有不等式约束
                Eigen::MatrixXd H = Bt.transpose() * Bt;
                Eigen::MatrixXd f = ((At.transpose() * Bt) + (Dt.transpose() * Bt) -
                                     (target.transpose() * Bt)).transpose();
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
                Eigen::MatrixXd u2 = Bt * Ut_1 - target + At + Dt;
                Ut = Ut_1 - (Bt.transpose() * u1 * u2);
                cout << "p < m,无约束，拉格朗日乘子法========================================================" << endl;
            } else {
                // 二次规划求解,小数点后第二位之后，对不上
                Eigen::MatrixXd H = Eigen::MatrixXd::Identity(m, m);
                Eigen::MatrixXd f = -Ut_1;
                Eigen::MatrixXd L = Eigen::MatrixXd::Identity(m, m);
                Eigen::MatrixXd k = ub;
                Eigen::MatrixXd Aeq = Bt;
                Eigen::MatrixXd beq = target - At - Dt;
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
    map <string, Eigen::MatrixXd> run() {
        map <string, Eigen::MatrixXd> result;
        Eigen::MatrixXd None;
        if (control == "intercept" || control == "i") {
            At = intercept_update();
            Dt = drift_update();
            Ut = cal_recipe(At, B0, Dt);

            result["recipe"] = Ut;
            result["At"] = At;
            result["Bt"] = None;
            result["Dt"] = Dt;
        } else {
            Bt = slope_update();
            Dt = drift_update();
            Ut = cal_recipe(A0, Bt, Dt);

            result["recipe"] = Ut;
            result["At"] = None;
            result["Bt"] = Bt;
            result["Dt"] = Dt;
        }
        return result;
    }

};


// 2. 测试案例
void test_rdewma() {




    // 配置Ut_1,Yt矩阵数组
    Eigen::MatrixXd Ut_1(2, 1); // 前1次的输入
    Eigen::MatrixXd Yt(2, 1); // 前1次的量测值
    Ut_1 << -0.622, -1.421;
    Yt << 1.047, 1.453;


    // 配置At_1,Bt_1,Dt_1矩阵数组
    Eigen::MatrixXd At_1(2, 1);
    Eigen::MatrixXd Bt_1(1, 2); // 注意此处的维度
    Eigen::MatrixXd Dt_1(2, 1);
    At_1 << 0.363, 0.643;
    Bt_1 << 0.363, 0.643;
    Dt_1 << 0.544, 0.965;


    // 配置A0,B0矩阵
    Eigen::MatrixXd A0{
            {1.0},
            {0.3}
    };
    Eigen::MatrixXd B0{
            {1.0, 0.2},
            {0.3, 1.0}
    };
    Eigen::MatrixXd D0;


    // 配置目标矩阵
    Eigen::VectorXd target(2);  // 期望目标值
    target << 0, 0;

    // 配置约束
    Eigen::VectorXd lb(2); // 输入的下界
    Eigen::VectorXd ub(2); // 输入的上界
    lb << -100, -100;
    ub << 100, 100;

    string control = "s";  // 可选截距更新或者斜率更新
    bool decay = false;
    int t = 0;

    // 配置其他参数
    double lamda1 = 0.2;  // 当前截距/斜率的权重因子
    double lamda2 = 0.3;


    // 运行
    RecursionDEWMA recursiondewma(Ut_1, Yt, At_1, Bt_1, Dt_1, A0, B0, D0, target, lb, ub, control, decay, t,
                                  lamda1, lamda2); //必须全部输入
    map <string, Eigen::MatrixXd> result = recursiondewma.run();
    cout << "********************************* result ***************************************" << endl;
    cout << "recipe：" << endl << result["recipe"] << endl;
    cout << "At：" << endl << result["At"] << endl;
    cout << "Bt：" << endl << result["Bt"] << endl;
    cout << "Dt：" << endl << result["Dt"] << endl;
};