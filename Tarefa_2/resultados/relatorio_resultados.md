# Relatório — Classificação de Imagens Territoriais com CNNs

## Metodologia

Foram comparadas duas redes convolucionais treinadas do zero: uma CNN simples, sem regularização explícita, e uma CNN regularizada com Batch Normalization e Dropout. Ambas receberam o mesmo pré-processamento, a mesma divisão estratificada, a mesma semente aleatória, o otimizador Adam e o mesmo número de épocas. Não foi usado aumento de dados, para isolar o efeito das técnicas solicitadas.

O experimento utilizou 11.228 imagens no treino, 2.806 na validação e 3.000 no teste, distribuídas nas classes buildings, forest, glacier, mountain, sea e street.

## Resultados no conjunto de teste

| Modelo | Acurácia | Precisão macro | Recall macro | F1 macro | Melhor validação | Tempo |
|---|---:|---:|---:|---:|---:|---:|
| CNN simples | 84.93% | 85.55% | 85.21% | 0.8507 | 85.35% | 8.1 min |
| CNN regularizada | 83.60% | 83.81% | 83.79% | 0.8374 | 84.07% | 7.1 min |

## Análise

- **CNN simples:** melhor validação na época 8; diferença treino–validação de +2.09% nessa época. Maior recall em `forest` (95.99%) e menor em `glacier` (70.16%).
- **CNN regularizada:** melhor validação na época 9; diferença treino–validação de -4.59% nessa época. Maior recall em `forest` (96.62%) e menor em `glacier` (75.95%).

## Conclusão

A **CNN simples** obteve o melhor resultado neste orçamento de treinamento, superando a CNN regularizada em 1.33% ponto percentual de acurácia e 0.0133 de F1 macro.

As curvas mostram que a CNN simples aprende mais rapidamente, mas começa a abrir uma diferença maior entre treino e validação nas épocas finais. A CNN regularizada mantém a acurácia de treino abaixo da validação porque o Dropout atua somente no treinamento; isso não é erro, e sim o comportamento esperado. Batch Normalization e Dropout controlaram a memorização, porém também tornaram a convergência mais lenta. Com apenas dez épocas, essa maior capacidade de generalização não se converteu em pontuação superior no teste. Mais épocas e ajuste das taxas de Dropout seriam uma extensão razoável, sem alterar a conclusão deste experimento.
