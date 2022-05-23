#ifndef EXTRAERDATA_H
#define EXTRAERDATA_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>
#include <stdlib.h>
#include <fstream>

/* En la clase se implementa el constructor, que recibe los argumentos de entrada
 * de la clase */

class ExtraerData
{
    /* Recibe el nombre del fichero CSV */
    std::string setDatos;
    /* Recibe el separador o delimitador */
    std::string delimitador;
    /*Recibe si tiene cabecera el fichero de datos */
    bool header;

public:
    ExtraerData(std::string datos, std::string separador, bool head):
    setDatos(datos),
    delimitador(separador),
    header(head) {}

    /* Prototipo de funciones */
    std::vector<std::vector<std::string>> ReadCSV();
    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int filas, int columnas);
    auto Promedio(Eigen::MatrixXd datos) -> decltype (datos.colwise().mean());
    auto DesvStand(Eigen::MatrixXd data) -> decltype (((data.array().square().colwise().sum())/(data.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd dataNorm, float sizeTrain);
    void ConVectorFichero(std::vector<float> vectorDatos, std::string fileName);
    void EigenToFile(Eigen::MatrixXd matrixData, std::string fileName);

};

#endif // EXTRAERDATA_H
