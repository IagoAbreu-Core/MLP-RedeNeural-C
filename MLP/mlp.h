#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#ifdef BIT32
    typedef float real;
#else
    typedef double real;
#endif


typedef struct {
    int entradas, saidas;
    char t;
    real **peso;
    real *bia;
    real *saida;
} Camada;

typedef struct {
    int num_camadas;
    int epoca;
    Camada **camadas;
} Rede_neural;

void definir_semente(uint32_t semente);

Rede_neural* criar_rede();

Camada* criar_camada(Rede_neural*,int,int,char);

void feedforward(Rede_neural*,real*);

void Print_saidas(Rede_neural*);

void backpropagation(Rede_neural*,double**,double,real*,real*);

void clear_rede(Rede_neural*);

double** criar_delta(Rede_neural*);

void clear_delta(double**,Rede_neural*);

void salva_modelo(Rede_neural*,char*);

void carrega_modelo(Rede_neural*,char*);

void salva_loss(Rede_neural* rede,real* alvo);

#endif /*MPL.H*/