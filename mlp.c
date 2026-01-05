#include "mlp.h"

double valor_random() {
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
}

double sigmoid(double y) {
    return 1.0 / (1.0 + exp(-y));
}

double ReLu(double y) {
    return (y >= 0) ? 1.0 : 0.0;
}

double d_sigmoid(double y) {
    return y * (1.0 - y);
}

double d_ReLu(double y) {
    return (y >= 0) ? 1.0 : 0.0;
}

Camada* criar_camada(Rede_neural* r, int x, int y, char t) {
    Camada *c = (Camada*)malloc(sizeof(Camada));
    c->entradas = x;
    c->saidas = y;
    c->t = t;

    c->bia = (double*)malloc(y * sizeof(double));
    c->saida = (double*)malloc(y * sizeof(double));
    c->peso = (double**)malloc(y * sizeof(double*));

    for(int i = 0; i < y; i++) {
        c->peso[i] = (double*)malloc(x * sizeof(double));
        for (int j = 0; j < x; j++)
            c->peso[i][j] = valor_random();
        c->bia[i] = valor_random();
    }

    r->num_camadas++;
    r->camadas = (Camada**)realloc(r->camadas, r->num_camadas * sizeof(Camada*));
    r->camadas[r->num_camadas - 1] = c;
    return c;
}

void feedforward(Rede_neural *rede, double *input) {
    double* entrada_atual = input;
    for(int i = 0; i < rede->num_camadas; i++) {
        Camada *c = rede->camadas[i];

        for(int j = 0; j < c->saidas; j++) {
            double soma = 0;
            for (int k = 0; k < c->entradas; k++)
                soma += entrada_atual[k] * c->peso[j][k];
            soma += c->bia[j];
            
            switch (c->t) {
                case 's' :
                    c->saida[j] = sigmoid(soma);
                    break;
                case 'r' :
                    c->saida[j] = ReLu(soma);
                    break;
                default :
                    printf("Ativação invalidade!\n");
                    return;
            }
        }
        entrada_atual = c->saida;
    }
}

void Print_saidas(Rede_neural* rede) {
    for(int i = 0;i < rede->camadas[rede->num_camadas - 1]->saidas;i++)
        printf("Saída [%d]: %f\n", i+1,rede->camadas[rede->num_camadas - 1]->saida[i]);
}

void backpropagation(Rede_neural* rede,double** delta, double* n, double* entrada_original, double* alvo) {
    if(delta != NULL && rede != NULL) {
        // 1. Calcular Deltas da Camada de Saída
        Camada* saida = rede->camadas[rede->num_camadas - 1];

        for(int i = 0; i < saida->saidas; i++) {
            double erro = alvo[i] - saida->saida[i];
            double valor;
            switch(saida->t) {
                case 's' :
                    valor = d_sigmoid(saida->saida[i]);
                    break;
                case 'r' :
                    valor = d_ReLu(saida->saida[i]);
                    break;
                default :
                    printf("Ativação invalidade!\n");
                    break;
            }
            delta[rede->num_camadas - 1][i] = erro * valor;
        }

        // 2. Calcular Deltas das Camadas Ocultas (de trás pra frente)
        for(int i = rede->num_camadas - 2; i >= 0; i--) {
            Camada* atual = rede->camadas[i];
            Camada* proxima = rede->camadas[i + 1];

            for(int j = 0; j < atual->saidas; j++) {
                double erro = 0.0;
                double valor;
                // Somatório do erro vindo da camada da frente
                for(int k = 0; k < proxima->saidas; k++) {
                    erro += delta[i+1][k] * proxima->peso[k][j];
                }
                switch(atual->t) {
                    case 's' :
                        valor = d_sigmoid(atual->saida[j]);
                        break;
                    case 'r' :
                        valor = d_ReLu(atual->saida[j]);
                        break;
                    default :
                        printf("Ativação invalidade!\n");
                        break;
                }
                delta[i][j] = erro * valor;
            }
        }

        // 3. Atualizar Pesos e Bias
        for(int i = 0; i < rede->num_camadas; i++) {
            Camada* c = rede->camadas[i];
            double* input = (i == 0) ? entrada_original : rede->camadas[i-1]->saida;

            for(int j = 0; j < c->saidas; j++) {
                for(int k = 0; k < c->entradas; k++) {
                    // Peso Novo = Peso Velho + (Taxa * Delta * Entrada)
                    c->peso[j][k] += *n * delta[i][j] * input[k];
                }
                c->bia[j] += *n * delta[i][j];
            }
        }
    }else {
        printf("Erro no Backpropagation!!\n");
    }
}

void clear_rede(Rede_neural *rede) {
    if(!rede) return;

    for(int i = 0; i < rede->num_camadas; i++) {
        Camada* c = rede->camadas[i];
        if(c) {
            for (int j = 0; j < c->saidas; j++) {
                free(c->peso[j]); // Libera o array interno de pesos
            }
            free(c->peso);     // Libera o array de ponteiros de pesos
            free(c->bia);
            free(c->saida);
            free(c);           // Libera a struct Camada
        }
    }
    free(rede->camadas); // Libera o array de ponteiros de Camada*
    free(rede);          // Libera a struct Rede_neural
}

double** criar_delta(Rede_neural* rede) {

    double** delta = (double**)malloc(rede->num_camadas * sizeof(double*));

    for(int i = 0; i < rede->num_camadas; i++) {
        delta[i] = (double*)malloc(rede->camadas[i]->saidas * sizeof(double));
    }

    return delta;
}

void clear_delta(double** delta,Rede_neural* rede) {
    if(delta != NULL){
        for(int i = 0; i < rede->num_camadas; i++) {
            free(delta[i]); 
        }
        free(delta);
    }
}

void salva_modelo(Rede_neural* rede,char *nome) {
    FILE* arq = fopen(nome, "wb");
    if(arq == NULL) {
        printf("Erro ao abrir o arquivo %s\n",nome);
        return;
    }

    fwrite(&(rede->num_camadas),sizeof(int), 1, arq);

    for(int i = 0;i < rede->num_camadas;i++) {
        Camada* c = rede->camadas[i];
        fwrite(&(c->entradas),sizeof(int), 1, arq);
        fwrite(&(c->saidas),sizeof(int), 1, arq);
        fwrite(&(c->t),sizeof(char), 1, arq);
        
        for(int j = 0;j < c->saidas;j++) {
            fwrite(c->peso[j],sizeof(double), c->entradas, arq);
        }

        fwrite(c->bia,sizeof(double), c->saidas, arq);
    }

    fclose(arq);
}

void carrega_modelo(Rede_neural* rede,char *nome) {
    FILE* arq = fopen(nome, "rb");

    if(arq == NULL) {
        printf("Erro ao abrir o arquivo %s\n",nome);
        return;
    }

    fread(&(rede->num_camadas),sizeof(int), 1, arq);
    rede->camadas = (Camada**)realloc(rede->camadas, rede->num_camadas * sizeof(Camada*));

    for(int i = 0;i < rede->num_camadas;i++) {
        Camada* c = (Camada*)malloc(sizeof(Camada));
        rede->camadas[i] = c;
        fread(&(c->entradas),sizeof(int), 1, arq);
        fread(&(c->saidas),sizeof(int), 1, arq);
        fread(&(c->t),sizeof(char), 1, arq);
        c->saida = (double*)malloc(c->saidas * sizeof(double));
        c->bia = (double*)malloc(c->saidas * sizeof(double));
        c->peso = (double**)malloc(c->saidas * sizeof(double*));

        for(int j = 0;j < c->saidas;j++) {
            c->peso[j] = (double*)malloc(c->entradas * sizeof(double));
            fread(c->peso[j],sizeof(double), c->entradas, arq);
        }

        fread(c->bia,sizeof(double), c->saidas, arq);
    }

    fclose(arq);
}