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
 * API para la lectura y manipulación de un fichero CSV´
 *****************************************************************************/
#include "extraerdata.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <stdlib.h>
#include <fstream>

/*****************************************************************************
 * Primer funcion miembro: lectura fichero CSV
 * vector de vectores "string". La idea es leer linea por linea y almacenar
 * en un vector de vectores de tipo "string".
 *****************************************************************************/
std::vector<std::vector<std::string>> ExtraerData::ReadCSV(){
    /* Se abre el fichro para lectura solamente */
    std::ifstream Fichero(setDatos);
    /* Vector de vectores "string": tendrá los datos del dataSet */
    std::vector<std::vector<std::string>> datosString;
    /* Se itera a traves de cada linea del dataSet, al tiempo que se divide
     * la linea con el delimitador */
    // Se almacena cada linea
    std::string linea = "";
    while(getline(Fichero, linea)){
        std::vector<std::string> vectorFila;
        // Dividimos según el delimitador
        boost::algorithm::split(vectorFila, linea, boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }
    /* Se cierra el fichero */
    Fichero.close();
    /* Se retorna el vector de vectores "string" */
    return datosString;
}

/* Segunda funcion para guardar el vector de vectores de tipo string: Se almacena
 * en una matriz para presentar como un objeto parecido al DATAFRAME que entrega
 * PANDAS. */
Eigen::MatrixXd ExtraerData::CSVtoEigen(std::vector<std::vector<std::string>> setDatos,
    int filas, int columnas){

    /* Si tiene cabecera se remueve, es decir, solo se manipulan datos */
    if(header == true){
        filas -=1;
    }

    /* Se itera sobre filas y columnas para lamacenar en la matriz de tamaño filas
     * x columnas. Basicamente, se almacenará "strings" en el vector: luego se pasan
     * a float para ser manipulados. */
    // Se crea la Matrix vacia
    Eigen::MatrixXd dfMatriz(columnas, filas);
    int i, j;
    for(i = 0; i < filas; i++){
        for(j = 0; j < columnas; j++){
            /* Con atof se pasan a tipo float los string por columna */
            dfMatriz(j,i) = atof(setDatos[i][j].c_str());
            //dfMatriz(j,i) = atof(setDatos([i][j].c_str());
        }
    }

    /* Se transpone la matriz para que sea filas x columnas y se devuelve o retorna
     * la matriz */

    return dfMatriz.transpose();
}

/* Se hace la funcion que retorne el promedio por cada dato (columna). La idea es
 * comparar con los realizado en python-pandas-sklearn para verificar que la función
 * artesanal corresponda (validar).
 * Auto y decltype: Especifica el tipo de la variable que se empieza a declarar,
 * la cual la deducirá de forma automatica con su inicializador (tiempo de compilación).
 * Para las funciones, si el tipo de retorno es un "auto", se evaluara mediante la
 * expresión del tipo de retorno en tiempo de compilación. */
auto ExtraerData::Promedio(Eigen::MatrixXd datos) -> decltype (datos.colwise().mean()){
    return datos.colwise().mean();
}

/* Función de Desciación Standar:
 * data = xi - x.promedio */

auto ExtraerData::DesvStand(Eigen::MatrixXd data) -> decltype (((data.array().square().colwise().sum())/(data.rows()-1)).sqrt()){
    return (((data.array().square().colwise().sum())/(data.rows()-1)).sqrt());
}

/* Acto seguido se procede a hacer el cálculo o la función de normalización: La idea
 * es vitar los cambios en orden de magnitud. Lo anterior representa un deterioro para 
 * la prediccion sobre la base de cualquier modelo de machine laearning. Evita los outliers */
Eigen::MatrixXd ExtraerData::Normalizador(Eigen::MatrixXd datos){
    /* Normalización:
     * MatrixNorm = xi - x.mean() / desviacionEstandar
    /* Primero se extrae el promedio */
//    auto PromedioD = Promedio(datos);
    /* Segundo se calcula la diferencia dato - promedio */
    Eigen::MatrixXd DataEscalado = datos.rowwise() - Promedio(datos);
    /* Tercero se calcula la desviación estandar */
//    auto DesviacionD = DesvStand(datos);
    Eigen::MatrixXd matrixNorm = DataEscalado.array().rowwise()/DesvStand(DataEscalado);
    /* Se retorna cada dato escalado */

    return matrixNorm;
}

/* Acontinuacion se implementa la función para la v¡división de los datos: Entrenamiento y prueba.
 * La idea es crear 4 matrices que tengan los datos de las variables dependientes e independientes
 * para el entrenamiento y las pruebas. Similar a la función con sklearn que seria: train_test_split(). */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ExtraerData::TrainTestSplit(Eigen::MatrixXd dataNorm, float sizeTrain){
    /* Numero de filas */
    int filas = dataNorm.rows();
    /* Numero de filas para entrenamiento */
    int filasTrain = round (sizeTrain * filas);
    /* Numero de filas para pruebas */
    int filasTest = filas - filasTrain;

    /* Con Eigen se puede especificar un bloque de una matriz superior o inferior a partir
     * de la fila que quieras como final o como principio el bloque. Para este caso en especial
     * se seleccionará como entrenamiento el bloque superior de la matriz dataNorm. Se deja entonces
     * la matriz inferior para prueba. */
    Eigen::MatrixXd train = dataNorm.topRows(filasTrain);

    /* Para este caso en especial (dataSet de entrada) se tiene que los datos de la columnas
     * se identifican en la parte izquierda las features o variables independientes quedando
     * la primera columna de la derecha como variable dependiente. */

    /* Se crea una matriz correspondiente a las features o variables independientes */
    Eigen::MatrixXd X_train = train.leftCols(dataNorm.cols()-1);

    /* Se crea una matriz correspondiente a la variable dependiente de la primera columna */
    Eigen::MatrixXd y_train = train.rightCols(1);

    Eigen::MatrixXd test = dataNorm.bottomRows(filasTest);
    /* Se hace el mismo procedimiento para los datos de prueba */
    Eigen::MatrixXd X_test = test.leftCols(dataNorm.cols()-1);
    Eigen::MatrixXd y_test = test.rightCols(1);

    /* Se retorna la tupla dada por el conjunto de datos de entrenamiento y prueba. Se
     * empaqueta la tupla con la funcion make_tuple, la cual debe ser desempaquetada en la
     * función principal. */
    return std::make_tuple(X_train, y_train, X_test, y_test);
}

/* A continusación se desarrollan dos funciones para la manipulación de vectores y la conversión
 * fichero a matriz Eigen. La manipulación de vectores representa iterar por el fichero de entrada
 * y convertirlo en vector de flotantes. */
void ExtraerData::ConVectorFichero(std::vector<float> vectorDatos, std::string fileName){
    /* Se crea un objeto que tendrá la lectura del fichero. */
    std::ofstream ficheroSalida(fileName);
    /* Se itera sobre el fichero de salida con el delimitador cambio de linea (\n), para ser
     * copiado en un vector (vectorSalida). */
    std::ostream_iterator<float> salidaIterador(ficheroSalida, "\n");
    /* Se copian los elementos del iterador en el vector de datos. */
    std::copy(vectorDatos.begin(), vectorDatos.end(), salidaIterador);
}

/* A continuación se desarrolla la función de conversión de matriz Eigen a fichero. Funcion util
 * dado que los valores parciales que se obtienen se imprimen en ficheros para tener seguridad
 * y trazabilidad. */
void ExtraerData::EigenToFile(Eigen::MatrixXd matrixData, std::string fileName){
    /* Se crea un objeto que tendrá la lectura del fichero. */
    std::ofstream ficheroSalida(fileName);
    if(ficheroSalida.is_open()){
        ficheroSalida << matrixData << "\n";
    }
}





















/*
ExtraerData::ExtraerData()
{

}*/
