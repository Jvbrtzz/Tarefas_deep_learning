# Predicao de churn com rede neural MLP

Projeto completo usando o dataset **IBM Telco Customer Churn**. O notebook cobre exploracao, limpeza, separacao treino/teste, codificacao de variaveis categoricas, padronizacao, treinamento de uma MLP, avaliacao e diagnostico de overfitting/underfitting.

## Arquivos principais

- `Predicao_Churn_MLP.ipynb`: atividade completa, ja executada e com os resultados visiveis.
- `data/Telco-Customer-Churn.csv`: dataset original.
- `outputs/modelo_churn_mlp.joblib`: pipeline completo treinado (pre-processamento + MLP).
- `outputs/metricas.csv`: metricas de treino, teste e baseline.
- `outputs/previsoes_teste.csv`: valores reais, probabilidades e classes previstas no teste.
- `requirements.txt`: dependencias com versoes usadas na validacao.

## Como executar novamente

Abra `Predicao_Churn_MLP.ipynb` no VS Code ou Jupyter, selecione o interpretador
`.venv\Scripts\python.exe` e use **Run All**. Se for necessario recriar o ambiente:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O notebook usa `random_state=42`, o que torna a divisao dos dados e a inicializacao da rede reproduziveis neste ambiente.

