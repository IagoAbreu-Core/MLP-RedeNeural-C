#include "./MLP/mlp.h"

int main() {
    definir_semente(time(NULL));
    
    Rede_neural* rede = criar_rede();
    double taxa_aprendizado = 0.1;
    double** delta = NULL;
    char* nome_arquivo = "modelo_xor.bin";

    real entrada[4][2] = {{0.0, 0.0},{1.0,0.0},{0.0,1.0},{1.0,1.0}};
    real alvo[4][1] = {{0},{1},{1},{0}};

    FILE* file = fopen(nome_arquivo,"rb");
    if(file != NULL) {
        fclose(file);
        carrega_modelo(rede,nome_arquivo);
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

    int max_epoca = 10000 + rede->epoca;
    for(rede->epoca; rede->epoca < max_epoca; rede->epoca++) { 
        int idx = rand() % 4;
        feedforward(rede, entrada[idx]);
        backpropagation(rede, delta, taxa_aprendizado, entrada[idx], alvo[idx]);
        if(rede->epoca%100 == 0)
            salva_loss(rede,alvo[idx]);
    }
    printf("\nModelo treinado..\n\n");

    for(int i=0; i < 4; i++) {
        feedforward(rede, entrada[i]);
        Print_saidas(rede);
    }

    salva_modelo(rede,nome_arquivo);
    clear_delta(delta,rede);
    clear_rede(rede);
    printf("\nFim do processo\n");

    return 0;
}