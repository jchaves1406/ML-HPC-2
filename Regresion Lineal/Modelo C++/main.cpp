/*****************************************************************************
 * Universidad Sergio Arboleda
 * Programa CC-IA
 * Materia: HPC2 - Mètricas
 * Autor: Jesus Chaves
 * Fecha: 28/02/2022
 * Tema: Introducción ML
 * Objetivo: Funcion principal para el calculo del modelo de regresion lineal.
 *
 * Requerimientos:
 * 1. - Aplicación o clase para la lectura de ficheros (csv), presente en una
 * clase, para la extracción de los datos, la normalizacion de los datos, en
 * general para la manipulacion de los datos.
 * 2. - Crear una clase para el calculo de la regresión lineal.
 *****************************************************************************/

#include "ExtraerData/extraerdata.h"
#include "linealregression.h"

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <string>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <stdlib.h>



int main(int argc, char *argv[]){
    /* Se crea un objeto del tipo extraerdata */
    ExtraerData extraer(argv[1],argv[2],argv[3]);

    /*Se crea el objeto de tipo LinealRegression */
    LinealRegression LR;

    /* Leer los datos del fichero por la funcion ReadCSV() del objeto extraer. */
    std::vector<std::vector<std::string>> DataFrame = extraer.ReadCSV();

    /* Para probar la función EigentoFile, y de esa manera imprimir el fichero de
     * datos, se debe definir el numero de filas y columnas del dataset. Basado
     * en los argumentos de entrada */
    int filas = DataFrame.size() + 1;
    int columnas = DataFrame[0].size();

    Eigen::MatrixXd MatDataFrame = extraer.CSVtoEigen(DataFrame, filas, columnas);

    /* Imprimir el objeto Matriz DataFrame */
//    std::cout << MatDataFrame << std::endl;

    /* Se imprime el vector de promedios por columna */
    std::cout << extraer.Promedio(MatDataFrame) << std::endl;

    std::cout << extraer.DesvStand(MatDataFrame) << std::endl;

    /* Se crea una matrix para almacenar la data normalizada */
    Eigen::MatrixXd dataNormalizada = extraer.Normalizador(MatDataFrame);
    /* Se imprimen los datos normalizados */
//    std::cout << dataNormalizada << std::endl;

    /* A coninuación se dividen en grupos de entrenamiento y pruebas la matriz dataNormalizada.
     * Se tomará para entrenamiento el 80% de los datos. */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> divData = extraer.TrainTestSplit(dataNormalizada, 0.8);
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd X_test;
    Eigen::MatrixXd y_test;
    std::tie(X_train, y_train, X_test, y_test) = divData;

    /* Se imprime numero de filas y columnas de cada uno de los elementos */
    std::cout << "Cantidad de registros matriz normalizada: " << dataNormalizada.rows() << std::endl;
    std::cout << "Cantidad de registros X_train: " << X_train.rows() << std::endl;
    std::cout << "Cantidad de registros y_train: " << y_train.rows() << std::endl;
    std::cout << "Cantidad de registros X_test: " << X_test.rows() << std::endl;
    std::cout << "Cantidad de registros y_test: " << y_test.rows() << std::endl;
    std::cout << "Cantidad de columnas X_train: " << X_train.cols() << std::endl;
    std::cout << "Cantidad de columnas y_train: " << y_train.cols() << std::endl;
    std::cout << "Cantidad de columnas X_test: " << X_test.cols() << std::endl;
    std::cout << "Cantidad de columnas y_test: " << y_test.cols() << std::endl;

    /* A continuación se desarrolla el primer algoritmo de machine learning Regresión Lineal.
     * Para el algoritmo se usa como ejemplo el dataSet de VinoRojo el cual tiene multiples variables.
     * Dada la naturaleza de RL, si se tienen variables con diferentes unidades (ordenes de magnitud)
     * una variable podria beneficiar/perjudicar otra variable: para ello se recomienda estandarizar
     * los datos, dando a todas las variables el mismo orden de magnitud y centradas en cero.
     * Es importante recalcar que la clase artesanal para la manipulación/tratamiento de datos, debe
     * ser observada de cerca a la hora de usar cualquier otro dataSet. */

    /* Se implementa el módulo de Machine Learning como clase de regresión lineal con su correspondiente
     * interfaz, se define el constructor y todas las funciones o métodos necesarios para la RL. Se tiene
     * en cuenta que la RL es un método estadistico que define la relación entre las variables independientes
     * y la variable dependiente. La idea principal es definir una linea recta. */

    /* A continuacion se define un vector para entrenamiento y prueba con valor inicial de 1 */
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

//    /* Se redimensionan las marices para ubicarlas en el vector de unos creado anteriormente. Similar a la
//     * función reshape de Numpy. */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

//    /* Se define el vector theta. */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());

//    /* Se define el vector alpha como ratio de aprendizaje (salto) */
    float alpha = 0.01;
    int iteraciones = 1000;

//    /* De igual forma se procedera a desempaquetar la tupla dada por el objeto (modelo). */
    std::tuple<Eigen::VectorXd, std::vector<float>> gradiente = LR.GradienteDescendiente(X_train, y_train, theta, alpha, iteraciones);

    Eigen::VectorXd thetaSalida;
    std::vector<float> costo;
    std::tie(thetaSalida, costo) = gradiente;

    /* Se imprime el vector de coeficientes o pesos. */
    std::cout << thetaSalida << std::endl;

    /* Se imprime el vector de costo para apreciar como decrementa su valor. */
//    for(auto v: costo){
//        std::cout << v << std::endl;
//    }

    /* Se exportan los valores de la función de costo y los coeficientes de theta a ficheros. */
    extraer.ConVectorFichero(costo,"VectorCosto.txt");
    extraer.EigenToFile(thetaSalida, "VectorTheta.txt");

    /* Se calcula de nuevo el promedio y la desviación estandar basada en los datos para calcular
     * y_hat (predicciones). */
    auto promedioData = extraer.Promedio(MatDataFrame);
    auto numFeatures = promedioData(0, 6);
    auto escalados = MatDataFrame.rowwise()-MatDataFrame.colwise().mean();
    auto sigmaData = extraer.DesvStand(escalados);
    auto sigmaFeatures = sigmaData(0, 6);

    Eigen::MatrixXd y_train_hat = (X_train * thetaSalida * sigmaFeatures).array() + numFeatures;
    Eigen::MatrixXd y_test_hat = (X_test * thetaSalida * sigmaFeatures).array() + numFeatures;
    Eigen::MatrixXd y_tr = MatDataFrame.col(6).topRows(1070);
    Eigen::MatrixXd y_ts = MatDataFrame.col(6).topRows(268);

//    /* A continuacion se determina que tan bueno es nuestro modelo. */
    float R2_train = LR.RSquared(y_tr, y_train_hat);
    float R2_test = LR.RSquared(y_ts, y_test_hat);

    std::cout << "Metrica R2 entrenamiento: " << R2_train << std::endl;
    std::cout << "Metrica R2 prueba: " << R2_test << std::endl;
//// Se exporta y_train_hat a fichero
    extraer.EigenToFile(y_train_hat, "y_train_hat.txt");
    extraer.EigenToFile(y_test_hat, "y_test_hat.txt");



    return EXIT_SUCCESS;
}
