library(tidyverse)

dados_loss <- read.csv("./dados/loss_total.csv")

plot <- ggplot(dados_loss, aes(x = Epoca, y = loss)) +
  geom_line(color = "blue", size = 1) +
  labs(
    title = "Curva de Aprendizado da Rede Neural MLP",
    subtitle = "Pesos treinados via código nativo em C",
    x = "Épocas de Treinamento",
    y = "Erro Quadrático Médio (Loss)"
  ) +
  theme_minimal()

ggsave("./graficos/curva_apredizado.pdf",width = 8, height = 6)
ggsave("./graficos/curva_apredizado.png",width = 8, height = 6)