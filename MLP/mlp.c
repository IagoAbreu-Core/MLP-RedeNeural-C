#include "mlp.h"

static uint32_t state = 40124917;
uint32_t xorshift() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void definir_semente(uint32_t semente) {
    if (semente != 0) state = semente;
}

real gerar_valor() {
    return ((real)xorshift() / 4294967295.0f) * 2.0f - 1.0f;
}

real sigmoid(real y) {
    return 1.0 / (1.0 + exp(-y));
}

real ReLu(real y) {
    return (y >= 0) ? 1.0 : 0.0;
}

real d_sigmoid(real y) {
    return y * (1.0 - y);
}

real d_ReLu(double y) {
    return (y >= 0) ? 1.0 : 0.0;
}

real d_tanh(real y) {
    return 1.0 - (y * y);
}

Rede_neural* criar_rede() {
    Rede_neural *rede = calloc(1, sizeof*rede);
    if (rede != NULL) {
        rede->num_camadas = 0;
        rede->epoca = 0;
        rede->camadas = NULL;
    }

    return rede;
}

Camada* criar_camada(Rede_neural* r, int x, int y, char t) {
    Camada *c = (Camada*)malloc(sizeof(Camada));
    c->entradas = x;
    c->saidas = y;
    c->t = t;

    c->bia = (real*)malloc(y * sizeof(real));
    c->saida = (real*)malloc(y * sizeof(real));
    c->peso = (real**)malloc(y * sizeof(real*));

    for(int i = 0; i < y; i++) {
        c->peso[i] = (real*)malloc(x * sizeof(real));
        for (int j = 0; j < x; j++)
            c->peso[i][j] = gerar_valor();
        c->bia[i] = gerar_valor();
    }

    r->num_camadas++;
    r->camadas = (Camada**)realloc(r->camadas, r->num_camadas * sizeof(Camada*));
    r->camadas[r->num_camadas - 1] = c;
    return c;
}

void feedforward(Rede_neural *rede, real *input) {
    real* entrada_atual = input;
    
    for(int i = 0; i < rede->num_camadas; i++) {
        Camada *c = rede->camadas[i];

        for(int j = 0; j < c->saidas; j++) {
            // 1. calcula a soma ponderada
            real soma = 0;
            for (int k = 0; k < c->entradas; k++) {
                soma += entrada_atual[k] * c->peso[j][k];
            }
            soma += c->bia[j];
            
            // 2. Aplica a Ativação
            switch (c->t) {
                case 's': c->saida[j] = sigmoid(soma); break;
                case 'r': c->saida[j] = ReLu(soma); break;
                case 't': c->saida[j] = (real)tanh(soma); break;
                default : printf("Ativação invalidade!\n");break;
            }

        }
        entrada_atual = c->saida;
    }
}

void Print_saidas(Rede_neural* rede) {
    for(int i = 0;i < rede->camadas[rede->num_camadas - 1]->saidas;i++)
        printf("Saída [%d]: %f\n", i+1,rede->camadas[rede->num_camadas - 1]->saida[i]);
}

void backpropagation(Rede_neural* rede,double** delta, double n, real* entrada_original, real* alvo) {
    // 1. Calcular Deltas da Camada de Saída
    Camada* saida = rede->camadas[rede->num_camadas - 1];

    for(int i = 0; i < saida->saidas; i++) {
        real erro = alvo[i] - saida->saida[i];
        real derivada;
        switch(saida->t) {
            case 's' :derivada = d_sigmoid(saida->saida[i]);break;
            case 'r' :derivada = d_ReLu(saida->saida[i]);break;
            case 't' :derivada = d_tanh(saida->saida[i]);break;
            default :printf("Ativação invalidade!\n");break;
        }
        delta[rede->num_camadas - 1][i] = erro * derivada;
    }

    // 2. Calcular Deltas das Camadas Ocultas
    for(int i = rede->num_camadas - 2; i >= 0; i--) {
        Camada* atual = rede->camadas[i];
        Camada* proxima = rede->camadas[i + 1];

        for(int j = 0; j < atual->saidas; j++) {

            real erro = 0.0;
            for(int k = 0; k < proxima->saidas; k++) {
                erro += delta[i+1][k] * proxima->peso[k][j];
            }

            real derivada;
            switch(atual->t) {
                case 's' :derivada = d_sigmoid(atual->saida[j]);break;
                case 'r' :derivada = d_ReLu(atual->saida[j]);break;
                case 't' :derivada = d_tanh(atual->saida[j]);break;
                default :printf("Ativação invalidade!\n");break;
            }
            delta[i][j] = erro * derivada;
        }
    }

    // 3. Atualizar Pesos
    for(int i = 0; i < rede->num_camadas; i++) {
        Camada* c = rede->camadas[i];
        real* input = (i == 0) ? entrada_original : rede->camadas[i-1]->saida;

        for(int j = 0; j < c->saidas; j++) {
            for(int k = 0; k < c->entradas; k++) {
                c->peso[j][k] += n * delta[i][j] * input[k];
            }
            c->bia[j] += n * delta[i][j];
            }
        }
}


void clear_rede(Rede_neural *rede) {
    if(!rede) return;

    for(int i = 0; i < rede->num_camadas; i++) {
        Camada* c = rede->camadas[i];
        if(c) {
            for (int j = 0; j < c->saidas; j++) {
                free(c->peso[j]);
            }
            free(c->peso);
            free(c->bia);
            free(c->saida);
            free(c);
        }
    }
    free(rede->camadas);
    free(rede);
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
    fwrite(&(rede->epoca),sizeof(int), 1,arq);

    for(int i = 0;i < rede->num_camadas;i++) {
        Camada* c = rede->camadas[i];
        fwrite(&(c->entradas),sizeof(int), 1, arq);
        fwrite(&(c->saidas),sizeof(int), 1, arq);
        fwrite(&(c->t),sizeof(char), 1, arq);
        
        for(int j = 0; j < c->saidas; j++)
            fwrite(c->peso[j], sizeof(real), c->entradas, arq);

	fwrite(c->bia, sizeof(real), c->saidas, arq);

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
    fread(&(rede->epoca),sizeof(int), 1,arq);
    rede->camadas = (Camada**)realloc(rede->camadas, rede->num_camadas * sizeof(Camada*));

    for(int i = 0;i < rede->num_camadas;i++) {
        Camada* c = (Camada*)malloc(sizeof(Camada));
        rede->camadas[i] = c;
        fread(&(c->entradas),sizeof(int), 1, arq);
        fread(&(c->saidas),sizeof(int), 1, arq);
        fread(&(c->t),sizeof(char), 1, arq);
        c->saida = (real*)malloc(c->saidas * sizeof(real));
        c->bia = (real*)malloc(c->saidas * sizeof(real));
        c->peso = (real**)malloc(c->saidas * sizeof(real*));

            for(int j = 0;j < c->saidas;j++) {
                c->peso[j] = (real*)malloc(c->entradas * sizeof(real));
                fread(c->peso[j],sizeof(real), c->entradas, arq);
            }

        fread(c->bia,sizeof(real), c->saidas, arq);
    }
    fclose(arq);
}

void salva_loss(Rede_neural* rede,real* alvo) {
    FILE* arq = fopen("loss_total.csv","r");

    if(arq == NULL) {
        arq = fopen("loss_total.csv","w");

        if(arq == NULL){
            printf("Erro ao criar o arquivo loss_total.csv\n");
            return;
        }

        fprintf(arq,"Saida,Epoca,loss\n");
    }else{
        fclose(arq);
        arq = fopen("loss_total.csv","a");

        if(arq == NULL){
            printf("Erro ao abrir o arquivo loss_total.csv\n");
            return;
        }
    }

    for(int i = 0;i < rede->camadas[rede->num_camadas - 1]->saidas;i++){
        real erro = alvo[i] - rede->camadas[rede->num_camadas - 1]->saida[i];
        real erro_q = erro * erro;
        char dado[50];
        snprintf(dado,sizeof(dado), "%d,%d,%lf\n",i+1,rede->epoca,erro_q);
        fprintf(arq,dado);
    }

    fclose(arq);
    
}