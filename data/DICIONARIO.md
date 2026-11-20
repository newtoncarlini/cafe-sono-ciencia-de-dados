# Dicionário e proveniência

Descrições fornecidas no enunciado do Checkpoint Cientista de Dados — Nível 1. Cada registro representa um participante **sintético**; não há período de coleta informado.

| Coluna | Significado | Tipo/unidade |
| --- | --- | --- |
| ID | Identificador único, não usado como preditor | Inteiro |
| Age | Idade | Anos, 18–80 |
| Gender | Gênero informado | Male, Female, Other |
| Country | País | Categoria |
| Coffee_Intake | Consumo diário de café | Xícaras/dia |
| Caffeine_mg | Cafeína diária estimada | mg/dia |
| Sleep_Hours | Duração média do sono noturno | Horas/noite |
| Sleep_Quality | Qualidade do sono; alvo do modelo | Poor, Fair, Good, Excellent |
| BMI | Índice de massa corporal | kg/m² |
| Heart_Rate | Frequência cardíaca de repouso | bpm |
| Stress_Level | Estresse | Low, Medium, High |
| Physical_Activity_Hours | Atividade física semanal | Horas/semana |
| Health_Issues | Condições de saúde | None, Mild, Moderate, Severe |
| Occupation | Ocupação principal | Office, Healthcare, Student, Service, Other |
| Smoking | Fumante | 0=não, 1=sim |
| Alcohol_Consumption | Consumo de álcool | 0=não, 1=sim |

`None` é texto válido e não é dado nulo. O notebook apresenta os limites realmente observados, além das faixas do enunciado.

## Campos derivados

- `Coffee_Stress_Interaction`: Coffee_Intake × Stress_Level codificado como Low=0, Medium=1, High=2. Criado em `src/features.py`, sem usar o alvo.
- Faixas de café: até 2; mais de 2 a 4; mais de 4 xícaras/dia. Usadas somente na análise descritiva.
- Faixas de idade: 18–29, 30–44, 45–59 e 60–80 anos. Usadas somente na análise descritiva.

## CSV processado

Os campos `num__*` são padronizados com estatísticas do treino. Os campos `cat__*` são indicadores one-hot ajustados no treino. O arquivo inclui `ID_referencia`, `Sleep_Quality` e `particao`; são metadados, não preditores.

O pipeline salvo recebe as colunas **originais**, não esse CSV já codificado. A lista de entrada está em `models/metadata.json`.
