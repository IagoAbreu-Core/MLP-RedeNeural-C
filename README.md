# MLP (Perceptron Multicamadas) em C puro

Este projeto consiste na implementação de uma Rede Neural Artificial do tipo MLP (Multilayer Perceptron) desenvolvida inteiramente em linguagem C.

**Objetivo:** Compreender o funcionamento interno de uma rede neural, incluindo o processo de aprendizado de máquina (Machine Learning) e processamento de dados, utilizando apenas as bibliotecas padrão da linguagem, sem o auxílio de frameworks externos como TensorFlow ou Keras.

## Motivações do Projeto

* **Algoritmos:** Prática intensiva de alocação dinâmica de memória e manipulação de ponteiros.
* **Matemática:** Estudo prático de funções de ativação como Sigmoid e ReLu para introduzir não-linearidade à rede. E o treinamento via Gradiente Descendente onde a rede calcula o erro na saída e propaga esse erro de volta para ajustar os pesos (W) e os bias (b).
* **Linguagem C:** Escolha baseada no alto desempenho e na necessidade de entender como o software interage com o hardware "baixo nível".

## Exemplo de Uso

Esse exemplo demonstra como criar uma rede neural de 4 camadas (Entrada, duas Camadas Ocultas e Saída) para simular a lógica XOR

Nesta lógica, a saída deve ser verdadeira (próxima de 1) apenas quando as entradas forem diferentes entre si. Se as entradas forem iguais, a saída deve ser falsa (próxima de 0). É um problema clássico que exige que a rede aprenda padrões não-lineares.

**Código :**

```c
#include "mlp.h"
#include <stdio.h>

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

```

**Como complilar :**

```sh
gcc exemplo.c mlp.c -o exemplo -lm
```

**Resultado esperado :**

```
modelo criado..

Saída [1]: 0.621081
Saída [1]: 0.619625
Saída [1]: 0.633081
Saída [1]: 0.631508

Modelo treinado..

Saída [1]: 0.498943
Saída [1]: 0.506722
Saída [1]: 0.524622
Saída [1]: 0.519195

Fim do processo
```

**E se executar de novo e para carregar modelo anterio :**

```
modelo carregado..

Saída [1]: 0.498943
Saída [1]: 0.506722
Saída [1]: 0.524622
Saída [1]: 0.519195

Modelo treinado..

Saída [1]: 0.161620
Saída [1]: 0.827890
Saída [1]: 0.860489
Saída [1]: 0.166453

Fim do processo
```
## Funções

**criar_camada** : Cria uma camada para rede neural. O char serve para tipo de ativação como 's' para sigmoid.

**criar_delta** : Cria os deltas de cada neurônio. deve ser usado depois de criados as camadas desejadas.

**Print_saidas** : Imprime os valores das saídas dos neurônio da ultima camada.

**feedforward** : Aonde os dados fluem da camada entrada passa uma ou mais camadas ocultas até camada saída.

**backpropagation** : Função para treinar a rede neural,que ajusta os pesos e bias para minimizar o erro e aproximar saída desejada.

**salva_modelo** : Salva a rede neural em um arquivo formato binário.

**carrega_modelo** : Carrega a rede neural a partir do arquivo binário.

**clear_delta** : Libera memória usada pelos deltas da rede.

**clear_rede** : Libera memória para a estrutura da rede_neural.