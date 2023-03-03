#include"ewma.h"
#include"solver.h"

using namespace std;


// 1. LoopDEWMA子类
class LoopDEWMA : public LoopEWMA {
public:
    Eigen::Array<Eigen::MatrixXd, -1, -1> A;
    Eigen::Array<Eigen::MatrixXd, -1, -1> B;
    Eigen::MatrixXd D0;
    Eigen::MatrixXd Dt;
    double lamda2;

    // 构造函数
    LoopDEWMA(
            Eigen::Array<Eigen::MatrixXd, -1, -1> U_,
            Eigen::Array<Eigen::MatrixXd, -1, -1> Y_,
            Eigen::Array<Eigen::MatrixXd, -1, -1> A_,
            Eigen::Array<Eigen::MatrixXd, -1, -1> B_,
            Eigen::VectorXd target_,
            double lamda1_,
            double lamda2_,
            Eigen::MatrixXd A0_,
            Eigen::MatrixXd B0_,
            Eigen::MatrixXd D0_,
            string control_,
            string type_,
            bool decay_,
            int t_,
            Eigen::VectorXd lb_,
            Eigen::VectorXd ub_) : LoopEWMA(U_, Y_, target_, lamda1_, A0_, B0_, control_, type_, decay_, t_, lb_, ub_),
                                   A(A_), B(B_), D0(D0_), lamda2(lamda2_) {

        // 代码审查
        transform(control_.begin(), control_.end(), control_.begin(), ::tolower);
        if (control_ == "i" || control_ == "intercept") {
            if (A_.size() == 0) {
                throw "当control为'i'或'intercept'时，A不能为空";
            } else {
                cout << "请注意A应该为array数组" << endl;
            }
        } else {
            if (B_.size() == 0) {
                throw "当control为's'或'slope'时，B不能为空";
            } else {
                cout << "请注意B应该为array数组" << endl;
            }
        }
        if (D0_.size() == 1) {
            cout << "将D0构造为重复矩阵" << endl;
        }


    }


    // 函数：drift_update
    Eigen::MatrixXd drift_update() {
        int row = int(U.rows());
        Eigen::Array<Eigen::MatrixXd, -1, 1> data_list;
        data_list.resize(row, 1);
        if (control == "i" || control == "intercept") {
            for (int i = 0; i < row; i++) {
                data_list(i) = Y(i) - B0 * U(i) - A(i);
            }
        } else {
            for (int i = 0; i < row; i++) {
                data_list(i) = Y(i) - A0 -
                               Eigen::MatrixXd::Ones(Y(i).rows(), Y(i).cols()) * (B(i).reshaped(1, -1) * U(i))(0, 0);
            }
        }
        Dt = ewma(data_list, lamda2);
        return Dt;
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
                Eigen::MatrixXd u2 = Bt * U(U.rows() - 1) - target + At + Dt;
                Ut = U(U.rows() - 1) - (Bt.transpose() * u1 * u2);
                cout << "p < m,无约束，拉格朗日乘子法========================================================" << endl;
            } else {
                // 二次规划求解,小数点后第二位之后，对不上
                Eigen::MatrixXd H = Eigen::MatrixXd::Identity(m, m);
                Eigen::MatrixXd f = -U(U.rows() - 1);
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
void test_dewma() {

    // 定义None值
    Eigen::Array<Eigen::MatrixXd, 0, 0> None;


    // 配置U,Y,A,B矩阵数组
    Eigen::Array<Eigen::MatrixXd, 3, 1> U;
    Eigen::Array<Eigen::MatrixXd, 3, 1> Y;
    Eigen::Array<Eigen::MatrixXd, 3, 1> A;
    Eigen::Array<Eigen::MatrixXd, 3, 1> B;
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
    A(0) = Eigen::MatrixXd{{0},
                           {0}};
    A(1) = Eigen::MatrixXd{{0.363},
                           {0.643}};
    A(2) = Eigen::MatrixXd{{0.681},
                           {1.127}};
    B(0) = Eigen::MatrixXd{{0},
                           {0}};
    B(1) = Eigen::MatrixXd{{0.363},
                           {0.643}};
    B(2) = Eigen::MatrixXd{{0.681},
                           {1.127}};


    // 配置A0,B0,D0矩阵
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
    Eigen::VectorXd target(2);
    target << 0, 0;


    // 配置其他参数
    double lamda1 = 0.2;
    double lamda2 = 0.3;
    bool decay = false;


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
    LoopDEWMA loopdewma(U, Y, A, B, target, lamda1, lamda2, A0, B0, D0, control, type, decay, t, lb, ub); //必须全部输入
    map <string, Eigen::MatrixXd> result = loopdewma.run();
    cout << "********************************* result ***************************************" << endl;
    cout << "recipe：" << endl << result["recipe"] << endl;
    cout << "At：" << endl << result["At"] << endl;
    cout << "Bt：" << endl << result["Bt"] << endl;
    cout << "Dt：" << endl << result["Dt"] << endl;
};