#ifndef LINEALREGRESSION_H
#define LINEALREGRESSION_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <string>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <stdlib.h>

class LinealRegression
{
public:
    LinealRegression(){}
    float FunCostOLS(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y,
                                                                          Eigen::MatrixXd theta, float alpha, int iteraciones);
    float RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);

};

#endif // LINEALREGRESSION_H
