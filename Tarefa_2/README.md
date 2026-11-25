# Classificação de Imagens Territoriais com CNNs

Aplicação Python (sem notebook) que baixa o dataset **Intel Image Classification**, treina duas CNNs sob as mesmas condições e compara os resultados no conjunto de teste:

1. CNN simples, sem regularização explícita;
2. CNN regularizada com Batch Normalization e Dropout.

## Decisão técnica

As duas CNNs usam dois blocos convolucionais, as mesmas imagens, a mesma ordem
de lotes e o mesmo número de épocas. A versão regularizada acrescenta Batch
Normalization e Dropout. O script usa o treino e o teste fornecidos pelo
dataset. O teste é executado ao fim das épocas, com os pesos da última época.

## Instalação

No diretório raiz do projeto, ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale o PyTorch. Para a GPU NVIDIA deste computador, a distribuição CUDA 11.8 é uma opção compatível:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Depois instale as demais dependências:

```powershell
python -m pip install -r .\Tarefa_2\requirements.txt
```

## Uso

Execute o script a partir da raiz do projeto. Ele usa as imagens locais ou
baixa o dataset do Kaggle quando necessário.

```powershell
python .\Tarefa_2\classificacao_territorial.py
```

Por padrão, cada CNN treina por 10 épocas. Use `--epocas` para mudar esse valor
e `--dados` para apontar para outra pasta com `seg_train` e `seg_test`.
O terminal mostra a perda por época e compara acurácia e F1 macro no teste.

## Versão TensorFlow/Keras

`classificacao_tensorflow.py` treina as duas variantes com `model.fit()` e usa
as mesmas pastas de imagens. A partir da raiz do projeto, crie um ambiente
separado em um caminho curto. Isso evita o limite de caminho do Windows
durante a instalação do TensorFlow dentro do OneDrive.

```powershell
$tfAmbiente = Join-Path $env:LOCALAPPDATA 'cnn-tf-pos'
python -m venv $tfAmbiente
$tfPython = Join-Path $tfAmbiente 'Scripts\python.exe'
& $tfPython -m pip install "tensorflow>=2.21,<2.22" "scikit-learn>=1.4,<2" "kagglehub>=0.4.3,<2"
& $tfPython .\Tarefa_2\classificacao_tensorflow.py
```

As opções `--epocas` e `--dados` também estão disponíveis. No Windows nativo,
essa versão do TensorFlow roda na CPU.

## Experimento original

Na versão anterior, com três blocos convolucionais e seleção pela validação,
a CNN simples alcançou
**84,93% de acurácia** e **0,8507 de F1 macro**. A CNN regularizada alcançou
**83,60% de acurácia** e **0,8374 de F1 macro**. O relatório e os gráficos dessa
execução estão em [resultados](resultados/relatorio_resultados.md).
