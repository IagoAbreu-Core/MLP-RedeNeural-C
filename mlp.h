#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    int entradas, saidas;
    char t;
    double **peso;
    double *bia;
    double *saida;
} Camada;

typedef struct {
    int num_camadas;
    Camada **camadas;
} Rede_neural;

double valor_random();

double sigmoid(double);

double d_sigmoid(double);

Camada* criar_camada(Rede_neural*,int,int,char);

void feedforward(Rede_neural*,double*);

void Print_saidas(Rede_neural*);

void backpropagation(Rede_neural*,double**,double*,double*,double*);

void clear_rede(Rede_neural*);

double** criar_delta(Rede_neural*);

void clear_delta(double**,Rede_neural*);

void salva_modelo(Rede_neural*,char*);

void carrega_modelo(Rede_neural*,char*);

#endif /*MPL.H*/