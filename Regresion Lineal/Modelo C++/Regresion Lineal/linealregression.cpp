#include "linealregression.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <string>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <stdlib.h>

/* En esta calse de desarrolla la función OLS y Gradiente descendiente. Tal y como ha sido demostrado
 * en clase. */

/* Se necesita entrenar el modelo, lo que implica minimizar alguna función de costo (se selecciona OLS),
 * la idea es medir la presición de la función de hipotesis. La función de costo es a forma de penalizar
 * al modelo por cometer un error. Se implementa una función que retorna un flotante, que toma como entrada
 * el dataSet (x, y), junto con los coeficientes (m1, m2, ... mn, b). */


float LinealRegression::FunCostOLS(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){
    Eigen::MatrixXd diferencia = pow((X*theta-y).array(), 2);
    return (diferencia.sum()/(2*X.rows()));
}

/* Se necesita proveer al programa una función para dar al algoritmo un valor inicial para theta, el cual
 * cambiará iterativamente hasta que converja el valor al minimo de nuestra función de costo. Básicamente
 * describe el Gradiente Descendiente: la idea es calcular el gradiente para la función de costo, dado por
 * la derivada parcial de la función de costo. La función tendrá un alfa que representa el salto del gradiente.
 * Las entradas para la función serán X, y, theta y el numero de veces para actualizar theta hasta que la
 * función converja. */
std::tuple<Eigen::VectorXd, std::vector<float>> LinealRegression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y,
                                                                                        Eigen::MatrixXd theta, float alpha, int iteraciones){
    /* Se almacena de forma temporal los parametros de theta. */
    Eigen::MatrixXd tempTheta = theta;

    /* Se extrae la cantidad de parametros (m: Features). */
    int parametros = theta.rows();

    /* Se ubica el costo inicial, que se actualiza con cada paso y los nuevos pesos */
    std::vector<float> costo;
    costo.push_back(FunCostOLS(X, y, theta));

    /* Por cada iteración se calcula la función de error, que se usa para multiplicar cada dimensión o
     * feature y así almacenarlo en la variable temporal. Se actualiza theta y se calacula el nuevo valor de
     * la función de costo basada en el nuevo valor de theta. */
    for(int i = 0; i < iteraciones; ++i){
        Eigen::MatrixXd error = X*theta-y;
        for(int j = 0; j < parametros; ++j){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd valorTemp = error.cwiseProduct(X_i);
            tempTheta(j,0) = theta(j,0)-((alpha/X.rows())*valorTemp.sum());
        }
        theta = tempTheta;
        costo.push_back(FunCostOLS(X, y, theta));
    }

    /* Se empaqueta la tupla para ser entregada. */
    return std::make_tuple(theta, costo);
}

/* Para determinar que tan bueno es el modelo que se ha desarrollado, a continuación se presenta una función
 * como métrica de evaluación: R2. R2 representa una medida de que tan bueno es el modelo. */
float LinealRegression::RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){

    auto numerador = pow((y-y_hat).array(), 2).sum();
    auto denominador = pow(y.array()-y.mean(), 2).sum();

    return 1-(numerador/denominador);
}



























