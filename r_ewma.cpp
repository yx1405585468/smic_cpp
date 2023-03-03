#include"solver.h"


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
    int t = 0;
    int p;
    int m;
    int T = 15;
    double lamda1 = 0.2;
    bool decay = false;
    string type = "ewma";
    string control = "intercept";


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
    ) {


        // U、Y、target审查
        if (Ut_1_.size() == 0 || Yt_.size() == 0 || target_.size() == 0) {
            throw "Ut_1 或 Yt 或 target 不能为空！";
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
            if (B0_.size() == 0 || At_1_.size() == 0) {
                throw "当control为'i'或'intercept'时，B0或At_1不能为空";
            }
        } else {
            if (A0_.size() == 0 || Bt_1_.size() == 0) {
                throw "当control为's'或'slope'时，A0或Bt_1不能为空";
            } else if (A0_.size() != Yt_.size()) {
                throw "当control为's'或'slope'时，A0与Yt的尺寸必须一致";
            }
        }

        // lb、ub审查
        if (lb_.size() == 1 || ub_.size() == 1) {
            cout << "请根据U(0).size()的值，构造对应维度的lb|ub的同值向量或矩阵。" << endl;
        }


        // 参数初始化
        Ut_1 = Ut_1_;
        Yt = Yt_;
        At_1 = At_1_;
        Bt_1 = Bt_1_;
        A0 = A0_;
        B0 = B0_;
        target = target_;
        lamda1 = lamda1_;
        decay = decay_;
        t = t_;
        lb = lb_;
        ub = ub_;
        control = control_;
        p = int(Yt_.rows());
        m = int(Ut_1_.rows());

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
        double t_lamda;
        if (decay) {
            t_lamda = lamda1 * exp(-t / T);
        } else {
            t_lamda = lamda1;
        }
        Eigen::MatrixXd At = t_lamda * (Yt - B0 * Ut_1) + (1 - t_lamda) * At_1;
        return At;

    }


    // 函数：slope_update()
    Eigen::MatrixXd slope_update() {
        double t_lamda;
        if (decay) {
            t_lamda = lamda1 * exp(-t / T);
        } else {
            t_lamda = lamda1;
        }

        Eigen::MatrixXd B_1 = (Yt - A0) * Ut_1.transpose();

        Eigen::MatrixXd B_2 = (Ut_1 * Ut_1.transpose() + 0.001 * Eigen::MatrixXd::Identity(m, m)).inverse();

        Eigen::MatrixXd B_ = t_lamda * (B_1 * B_2);

        Eigen::MatrixXd C_ = (1 - t_lamda) * Bt_1;
        if (C_.size() == B_.size()) {
            Bt = B_ + C_;
        } else {
            if (B_.cols() != C_.cols()) {
                Eigen::MatrixXd CL(B_.rows(), B_.cols());
                for (int i = 0; i < B_.rows(); i++) {
                    for (int j = 0; j < B_.cols(); j++) {
                        CL(i, j) = C_(i, 0);
                    }
                }
                Bt = B_ + CL;
            } else {
                Eigen::MatrixXd CL(B_.rows(), B_.cols());
                for (int i = 0; i < B_.rows(); i++) {
                    for (int j = 0; j < B_.cols(); j++) {
                        CL(i, j) = C_(0, j);
                    }
                }
                Bt = B_ + CL;
            }


        }
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
                Eigen::MatrixXd u2 = Bt * Ut_1 - target + At;
                Ut = Ut_1 - (Bt.transpose() * u1 * u2);
                cout << "p < m,无约束，拉格朗日乘子法========================================================" << endl;
            } else {
                // 二次规划求解,小数点后第二位之后，对不上
                Eigen::MatrixXd H = Eigen::MatrixXd::Identity(m, m);
                Eigen::MatrixXd f = -Ut_1;
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
    map <string, Eigen::MatrixXd> run() {
        Eigen::MatrixXd None;
        map <string, Eigen::MatrixXd> result;
        if (control == "intercept" || control == "i") {
            At = intercept_update();
            Ut = cal_recipe(At, B0);
            result["recipe"] = Ut;
            result["At"] = At;
            result["Bt"] = None;
        } else {
            Bt = slope_update();
            Ut = cal_recipe(A0, Bt);
            result["recipe"] = Ut;
            result["At"] = None;
            result["Bt"] = Bt;
        }
        return result;
    }
};


// 2. 测试案例
void test_rewma() {

    // 定义None值
    Eigen::MatrixXd None;
    //cout << "向量V的None值：" << V.size() << endl; // 打印结果为0


    // 配置Ut_1,Yt矩阵数组
    Eigen::MatrixXd Ut_1(2, 1); // 前1次的输入
    Eigen::MatrixXd Yt(2, 1); // 前1次的量测值
    Ut_1 << -0.622, -1.421;
    Yt << 1.047, 1.453;


    // 配置At_1,Bt_1矩阵数组
    Eigen::MatrixXd At_1(2, 1); // 前1次的输入
    Eigen::MatrixXd Bt_1(2, 1); // 前1次的量测值
    At_1 << 0.363, 0.643;
    Bt_1 << 0.363, 0.643;


    // 配置A0,B0矩阵
    Eigen::MatrixXd A0{
            {1.0},
            {0.3}
    };
    Eigen::MatrixXd B0{
            {1.0, 0.2},
            {0.3, 1.0}
    };


    // 配置目标矩阵
    Eigen::VectorXd target(2);  // 期望目标值
    target << 0, 0;


    // 配置其他参数
    double lamda1 = 0.2;  // 当前截距/斜率的权重因子
    bool decay = false;  // 衰减权重因子


    // 配置运行次数
    int t = 0;


    // 配置约束
    Eigen::VectorXd lb(2); // 输入的下界
    Eigen::VectorXd ub(2); // 输入的上界
    lb << -100, -100;
    ub << 100, 100;


    // 配置更新方式和ewma类型
    string control = "s";  // 可选截距更新或者斜率更新


    // 运行
    RecursionEWMA recursionewma(Ut_1, Yt, target, lamda1, A0, B0, control, decay, t, lb, ub, At_1, Bt_1);
    map <string, Eigen::MatrixXd> result = recursionewma.run();
    cout << "********************************* result ***************************************" << endl;
    cout << "recipe：" << endl << result["recipe"] << endl;
    cout << "At：" << endl << result["At"] << endl;
    cout << "Bt：" << endl << result["Bt"] << endl;
};