#include "mlp.h"

int main() {

    Rede_neural *rede = malloc(sizeof(Rede_neural));
    double taxa_aprendizado = 0.1;
    double** delta = NULL;

    double entrada[4][2] = {{0.0, 0.0},{1.0,0.0},{0.0,1.0},{1.0,1.0}};
    double alvo[4][1] = {{0},{1},{1},{0}};

    if(fopen("modelo_1","rb") != NULL) {
        carrega_modelo(rede,"modelo_1");
        printf("modelo carregado..\n\n");
    }
    else {
        criar_camada(rede,2,11,'s');
        criar_camada(rede,11,6,'s');
        criar_camada(rede,6,1,'s');
        printf("modelo criado..\n\n");
    }

    delta = criar_delta(rede);

    for(int i=0; i < 4; i++) {
        feedforward(rede, entrada[i]);
        Print_saidas(rede);
    }

    for(int i=0; i < 10000; i++) { 
        int idx = rand() % 4;
        feedforward(rede, entrada[idx]);
        backpropagation(rede, delta, &taxa_aprendizado, entrada[idx], alvo[idx]);
    }
    printf("\nModelo treinado..\n\n");

    for(int i=0; i < 4; i++) {
        feedforward(rede, entrada[i]);
        Print_saidas(rede);
    }

    salva_modelo(rede,"modelo_1");
    clear_delta(delta,rede);
    clear_rede(rede);
    printf("\nFim do processo\n");

    return 0;
}