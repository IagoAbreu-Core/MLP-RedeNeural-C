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

**Estrutura da Rede:**

**Entrada:** 2 neurônios.

**Camada Oculta 1:** 11 neurônios.

**Camada Oculta 2:** 6 neurônios.

**Saída:** 1 neurônio.

**Código :**

```c
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

```

**Como complilar :**

```sh
gcc ./scripts/logica_xor.c ./MLP/mlp.c -o logica_xor -lm
```

**Resultado esperado :**

```
modelo criado..

Saída [1]: 0.516581
Saída [1]: 0.528567
Saída [1]: 0.541800
Saída [1]: 0.544568

Modelo treinado..

Saída [1]: 0.216683
Saída [1]: 0.692686
Saída [1]: 0.713926
Saída [1]: 0.534534

Fim do processo
```

**E se executar de novo e para carregar modelo anterio :**

```
modelo carregado..

Saída [1]: 0.216683
Saída [1]: 0.692686
Saída [1]: 0.713926
Saída [1]: 0.534534

Modelo treinado..

Saída [1]: 0.056023
Saída [1]: 0.939695
Saída [1]: 0.938853
Saída [1]: 0.075813

Fim do processo
```
# Curva de aprendizando
Fiz um gráfico simples em R para demonstração da curva de aprendizando da rede neural

![grafico](graficos/Rplot.png)

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

**salva_loss** : Salva os loss de cada saida em um arquivo csv para analisa.
