# Café, hábitos e qualidade do sono

Projeto do **Checkpoint Cientista de Dados — Nível 1 (Alura)**. Investiga associações entre consumo de café, estresse e sono em 10 mil registros sintéticos e compara modelos de classificação de `Sleep_Quality`.

## Abrir o trabalho

O [notebook executado](notebooks/01_cafe_sono.ipynb) contém a análise completa, com textos, tabelas e gráficos já visíveis sem executar. O [resumo dos resultados](reports/resultados.md) apresenta os achados calculados.

### Resultados principais

| Modelo | F1 macro na validação | F1 macro no teste | Acurácia no teste |
| --- | ---: | ---: | ---: |
| Referência majoritária | 0,1802 | 0,1803 | 56,40% |
| Regressão logística | 0,7459 | 0,7454 | 68,30% |
| Floresta aleatória | **0,7679** | **0,7676** | **76,35%** |

A floresta foi selecionada pela validação cruzada. Sem estresse e a interação associada, seu F1 macro cai para **0,2476**. Isso demonstra dependência da estrutura sintética dos dados, não uma previsão clínica validada. O notebook detalha o experimento e os erros por classe.

### Entregas

1. **EDA:** dimensões, tipos, nulos, duplicados, faixas, estatísticas, histogramas, boxplots, barras, pizza e correlações.
2. **Visualização e insights:** café e sono, estresse, diferenças por gênero, segmentação por idade e café e seção **Principais Descobertas**.
3. **Classificação:** feature derivada, pré-processamento em pipelines, divisão estratificada, referência majoritária, regressão logística e floresta aleatória, validação cruzada, teste reservado, métricas e matriz de confusão. Dataset processado e melhor pipeline salvos.

## Como reproduzir localmente

Ambiente validado: **Python 3.12**. As versões utilizadas estão em `requirements.txt`; não se assume compatibilidade com Python 3.8.

No terminal, dentro da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m jupyter notebook notebooks/01_cafe_sono.ipynb
```

Para executar tudo e salvar as saídas:

```powershell
.\.venv\Scripts\python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_cafe_sono.ipynb --ExecutePreprocessor.timeout=600
```

Execute as células do início ao fim. O CSV original acompanha o projeto; se ele for removido, o notebook tenta baixá-lo do endereço oficial informado. As pastas devem manter a organização abaixo, inclusive `src`, necessária para carregar o modelo.

## Organização

```text
notebooks/01_cafe_sono.ipynb       análise completa executada
src/features.py                  transformação reutilizável da feature derivada
src/predict.py                   exemplo de inferência em novos registros
data/raw/                        CSV original
data/processed/                  CSV numérico processado, alvo e partição
models/modelo_sono.joblib         pipeline selecionado, ajustado apenas no treino
models/metadata.json              colunas, métricas, versões e proveniência
reports/figures/                  gráficos da análise
reports/                         tabelas de auditoria, métricas e resultados
requirements.txt                 versões para reprodução
build_notebook.py                construção reproduzível do notebook
```

## Método e cuidados

- Separação **80% treino / 20% teste**, estratificada pelo alvo e com semente 42.
- Seleção por **F1 macro em validação cruzada de três dobras no treino**. O teste não decide o modelo.
- Imputação, escala, encoding e feature derivada ficam dentro do pipeline para evitar vazamento entre dobras.
- A feature `Coffee_Stress_Interaction` multiplica café por um código ordinal de estresse (0/1/2); é uma hipótese exploratória, não uma medida clínica.
- `None` em `Health_Issues` é uma categoria válida. O leitor do CSV preserva esse texto em vez de convertê-lo em ausência.
- `ID`, `Caffeine_mg`, `Sleep_Hours` e `Health_Issues` são excluídos do cenário principal. O notebook explica identificadores, redundância e disponibilidade temporal.
- Um experimento separado inclui `Sleep_Hours` para demonstrar o risco de reconstruir a regra do alvo. Ele não substitui o modelo principal.
- A auditoria também encontrou uma associação quase determinística entre estresse e alvo. Um cenário de sensibilidade remove estresse **e** a interação derivada; a diferença de desempenho é documentada, sem interpretar o cenário principal como validação prospectiva.
- O CSV processado contém `ID_referencia`, `Sleep_Quality` e `particao` como metadados; essas colunas **não** devem entrar como preditores.
- O modelo exportado preserva o ajuste no treino; não foi reajustado com o teste após a avaliação.

## Utilizar o modelo

Forneça as colunas originais de entrada listadas em `models/metadata.json`. O pipeline cria a feature derivada e codifica as categorias automaticamente:

```powershell
python -m src.predict --input data/raw/synthetic_coffee_health_10000.csv --output reports/exemplo_previsoes.csv
```

Carregue arquivos `joblib` apenas de fontes confiáveis e use o ambiente com as versões registradas. O exemplo usa o CSV existente apenas para demonstrar a interface; isso não é uma nova avaliação independente.

## Dados e limites de interpretação

- Fonte usada: [CSV fornecido pela Alura](https://cdn3.gnarususercontent.com.br/5209-projeto-cientista-dados-nivel-1/synthetic_coffee_health_10000%28in%29.csv).
- Fonte original indicada no enunciado: [Global Coffee Health Dataset — Kaggle](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset).
- O hash SHA-256 da cópia analisada está em `reports/proveniencia.json`.
- Os dados são **sintéticos**. Não há demonstração de representatividade, causalidade ou eficácia de intervenções.
- Os modelos são um exercício de classificação, **sem validação clínica ou indicação de uso em decisões de saúde**.
- A disponibilidade pública do download não substitui a licença da fonte. Não é atribuída uma nova licença ao dataset neste projeto; preserve a atribuição e confira os termos da origem antes de redistribuir.
