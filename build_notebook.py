"""Constrói o notebook de análise; a execução é feita com nbclient."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# Café, hábitos e qualidade do sono
## Checkpoint Cientista de Dados — Nível 1

Projeto de análise exploratória, visualização e classificação para a empresa fictícia **Health&Life Analytics**.
Base sintética disponibilizada pela Alura: 10 mil registros de participantes simulados.
Os resultados descrevem esta base e não demonstram efeitos causais nem orientações clínicas.

## Resumo
O resumo com resultados é calculado ao final da execução e também salvo em `reports/resultados.md`.
Este notebook contém as três entregas, as verificações de qualidade e os arquivos para reprodução.

## Contexto e método
Investigar associações entre consumo de café e sono e prever `Sleep_Quality` a partir dos hábitos disponíveis.
Cada linha é um participante sintético; não há datas de coleta, acompanhamento longitudinal ou evidência de amostragem real.

### Premissas e prevenção de vazamento
- Não interpretar associação como causa.
- Preservar a categoria literal `None` em `Health_Issues`.
- Excluir `ID`, pois é um identificador.
- Excluir `Caffeine_mg` do modelo principal: mede praticamente o mesmo consumo que `Coffee_Intake`.
- Excluir `Sleep_Hours` do modelo principal: é uma medida simultânea ao desfecho e pode revelar sua construção sintética.
- Excluir `Health_Issues` do modelo principal: não sabemos se foi observado antes ou depois do sono.
- Usar divisão estratificada 80/20, com semente 42; selecionar modelo apenas por validação cruzada no treino.
- Comparar uma referência de classe majoritária, regressão logística e floresta aleatória.
- Priorizar F1 macro para dar o mesmo peso às quatro classes, sem ignorar acurácia e métricas por classe.
- Executar um cenário diagnóstico separado com `Sleep_Hours`, sem usá-lo para selecionar o modelo principal.
''')
code('''from pathlib import Path
import sys, json, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import joblib

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
assert (ROOT / 'src/features.py').exists(), 'Abra o notebook dentro da pasta do projeto.'
sys.path.insert(0, str(ROOT))
from src.features import HabitFeatures
SEED = 42
SOURCE = 'https://cdn3.gnarususercontent.com.br/5209-projeto-cientista-dados-nivel-1/synthetic_coffee_health_10000%28in%29.csv'
for folder in ['data/raw', 'data/processed', 'models', 'reports/figures']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
plt.rcParams.update({'figure.figsize': (10, 5), 'axes.titlesize': 14, 'axes.labelsize': 11,
                     'savefig.dpi': 150, 'figure.dpi': 100})
ORDER = ['Poor', 'Fair', 'Good', 'Excellent']
def savefig(name):
    plt.tight_layout()
    plt.savefig(ROOT / 'reports/figures' / f'{name}.png', bbox_inches='tight')
    plt.show()
''')
md('''## Entrega 1 — Análise exploratória
### 1. Fonte, estrutura e qualidade
A cópia original do CSV é mantida sem alterações em `data/raw`. A categoria textual `None` significa ausência de problemas de saúde segundo o dicionário da atividade.
Fonte original indicada pela atividade: [Global Coffee Health Dataset](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset).
A redistribuição segue os termos da fonte; esta entrega não atribui uma nova licença aos dados.
''')
code('''path = ROOT / 'data/raw/synthetic_coffee_health_10000.csv'
if not path.exists():
    from urllib.request import urlretrieve
    urlretrieve(SOURCE, path)
dados = pd.read_csv(path, keep_default_na=False, na_values=['', 'NA', 'N/A', 'null'])
proveniencia = {'url': SOURCE, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'registros': len(dados), 'colunas': len(dados.columns), 'natureza': 'sintética'}
display(pd.Series(proveniencia, name='Fonte'))
display(dados.head())
qualidade = pd.DataFrame({'tipo': dados.dtypes.astype(str), 'nulos': dados.isna().sum(),
                         'valores_distintos': dados.nunique()})
display(qualidade)
print('Linhas duplicadas:', dados.duplicated().sum())
print('IDs duplicados:', dados.ID.duplicated().sum())
print('Duplicados sem considerar ID:', dados.drop(columns='ID').duplicated().sum())
assert dados.ID.is_unique and dados.Sleep_Quality.notna().all()
assert set(dados.Sleep_Quality) == set(ORDER)
print('Categoria None preservada:', (dados.Health_Issues == 'None').sum())
display(dados.describe().round(2))
display(dados.describe(include='object').T)
qualidade.to_csv(ROOT / 'reports/qualidade_dados.csv', index_label='coluna')
(ROOT / 'reports/proveniencia.json').write_text(json.dumps(proveniencia, indent=2, ensure_ascii=False), encoding='utf-8')
''')
code('''limites = {'Age': (18, 80), 'Coffee_Intake': (0, 10), 'Sleep_Hours': (3, 10),
           'BMI': (15, 40), 'Heart_Rate': (50, 110), 'Physical_Activity_Hours': (0, 15)}
auditoria = pd.DataFrame([
    {'variavel': c, 'minimo_observado': dados[c].min(), 'maximo_observado': dados[c].max(),
     'fora_faixa_dicionario': int((~dados[c].between(lo, hi)).sum())}
    for c, (lo, hi) in limites.items()])
display(auditoria)
display(pd.Series({c: sorted(dados[c].unique()) for c in ['Smoking', 'Alcohol_Consumption']}, name='valores').to_frame())
auditoria.to_csv(ROOT / 'reports/faixas_dados.csv', index=False)
display(Markdown('**Tratamento:** manter extremos plausíveis; nenhum outlier será removido automaticamente. '
                 'Valores fora das faixas documentadas são expostos acima, sem correções silenciosas.'))
''')
md('''### 2. Variáveis numéricas
Histogramas mostram o formato das distribuições; boxplots mostram mediana, quartis e extremos.
Pontos fora dos bigodes não são, por si só, erros de cadastro.
''')
code('''numericas = ['Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 'BMI', 'Heart_Rate', 'Physical_Activity_Hours']
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for ax, col in zip(axes.flat, numericas):
    sns.histplot(dados[col], bins=25, ax=ax, color='#287D8E')
    ax.set(title=col, xlabel=col, ylabel='Participantes (n)')
for ax in axes.flat[len(numericas):]: ax.set_visible(False)
fig.suptitle('Distribuições numéricas — 10.000 participantes sintéticos', y=1.01)
savefig('01_histogramas')
''')
code('''fig, axes = plt.subplots(4, 2, figsize=(13, 10))
for ax, col in zip(axes.flat, numericas):
    sns.boxplot(x=dados[col], ax=ax, color='#8AC6C0')
    ax.set(title=col, xlabel=col)
axes.flat[-1].set_visible(False)
fig.suptitle('Dispersão e extremos por variável', y=1.01)
savefig('02_boxplots')
''')
md('''### 3. Variáveis categóricas
As barras mostram contagens, incluindo ausência declarada de problemas de saúde.
O gráfico de pizza complementa a distribuição do alvo com percentuais e tamanho de cada classe.
''')
code('''categoricas = ['Gender', 'Country', 'Sleep_Quality', 'Stress_Level', 'Health_Issues', 'Occupation', 'Smoking', 'Alcohol_Consumption']
fig, axes = plt.subplots(4, 2, figsize=(15, 19))
for ax, col in zip(axes.flat, categoricas):
    counts = dados[col].value_counts().sort_values()
    ax.barh(counts.index.astype(str), counts.values, color='#287D8E')
    ax.set(title=col, xlabel='Participantes (n)')
    ax.bar_label(ax.containers[0], padding=3, fontsize=8)
    ax.set_xlim(0, counts.max()*1.18)
savefig('03_categoricas')
''')
code('''classes = dados.Sleep_Quality.value_counts().reindex(ORDER)
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(classes, labels=[f'{k} (n={v:,})' for k, v in classes.items()],
       autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
ax.set_title('Qualidade do sono — proporção de cada classe')
savefig('04_proporcao_alvo')
display(Markdown(f'A classe mais frequente é **{classes.idxmax()}**, com **{classes.max()/classes.sum():.1%}** dos registros. '
                 'Isso justifica comparar os modelos com uma referência majoritária e avaliar F1 macro.'))
''')
md('''### 4. Correlações e redundância
Pearson mede associação linear, não causalidade. O identificador é excluído do mapa.
Cafeína estimada e xícaras medem quase o mesmo hábito; tratá-las como evidências independentes exageraria a informação disponível.
''')
code('''corr = dados[numericas].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title('Correlações de Pearson — variáveis numéricas')
savefig('05_correlacoes')
display(Markdown(f'Correlação café–horas de sono: **{corr.loc["Coffee_Intake", "Sleep_Hours"]:.3f}**. '
                 f'Correlação café–cafeína: **{corr.loc["Coffee_Intake", "Caffeine_mg"]:.3f}**.'))
''')
md('''## Entrega 2 — Visualização e insights
### 5. Café, estresse e horas de sono
A amostra do gráfico de dispersão é determinística e usada apenas para legibilidade; as tabelas e demais gráficos usam todos os registros.
''')
code('''fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=dados.sample(min(1500, len(dados)), random_state=SEED), x='Coffee_Intake', y='Sleep_Hours',
                hue='Stress_Level', hue_order=['Low', 'Medium', 'High'], alpha=.4, s=20, ax=axes[0])
axes[0].set(title='Café e sono — amostra de 1.500 participantes', xlabel='Café (xícaras/dia)', ylabel='Sono (horas/noite)')
sns.boxplot(data=dados, x='Stress_Level', y='Sleep_Hours', order=['Low', 'Medium', 'High'], color='#8AC6C0', ax=axes[1])
axes[1].set(title='Sono por nível de estresse — base completa', xlabel='Estresse', ylabel='Sono (horas/noite)')
savefig('06_cafe_estresse_sono')
''')
code('''analise = dados.assign(
    Faixa_Cafe=pd.cut(dados.Coffee_Intake, [-.01, 2, 4, 10], labels=['Até 2', 'Mais de 2 a 4', 'Mais de 4']),
    Faixa_Idade=pd.cut(dados.Age, [17, 29, 44, 59, 80], labels=['18–29', '30–44', '45–59', '60–80']))
resumo_cafe = analise.groupby('Faixa_Cafe', observed=True).Sleep_Hours.agg(n='size', media='mean', mediana='median', desvio='std')
resumo_genero = analise.groupby('Gender').Sleep_Hours.agg(n='size', media='mean', desvio='std')
resumo_estresse = analise.groupby('Stress_Level').Sleep_Hours.agg(n='size', media='mean', desvio='std').reindex(['Low','Medium','High'])
display(resumo_cafe.round(3)); display(resumo_genero.round(3)); display(resumo_estresse.round(3))
for name, table in [('cafe', resumo_cafe), ('genero', resumo_genero), ('estresse', resumo_estresse)]:
    table.to_csv(ROOT / f'reports/sono_por_{name}.csv')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=analise, x='Gender', y='Sleep_Hours', color='#8AC6C0', ax=axes[0])
axes[0].set(title='Distribuição do sono por gênero', xlabel='Gênero', ylabel='Sono (horas/noite)')
proporcao = pd.crosstab(analise.Stress_Level, analise.Sleep_Quality, normalize='index').reindex(index=['Low','Medium','High'], columns=ORDER)
proporcao.plot.bar(stacked=True, ax=axes[1], rot=0)
axes[1].set(title='Qualidade do sono dentro de cada nível de estresse', xlabel='Estresse', ylabel='Proporção dos participantes', ylim=(0,1))
axes[1].legend(title='Qualidade', bbox_to_anchor=(1.02,1), loc='upper left')
savefig('07_genero_qualidade')
''')
code('''fig, axes = plt.subplots(1, 2, figsize=(14, 5))
medias = analise.pivot_table(index='Faixa_Idade', columns='Faixa_Cafe', values='Sleep_Hours', aggfunc='mean', observed=True)
contagens = analise.pivot_table(index='Faixa_Idade', columns='Faixa_Cafe', values='Sleep_Hours', aggfunc='size', observed=True)
sns.heatmap(medias, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], cbar_kws={'label':'Horas/noite'})
axes[0].set(title='Sono médio por idade e café', xlabel='Café (xícaras/dia)', ylabel='Idade (anos)')
sns.heatmap(contagens, annot=True, fmt='d', cmap='Greys', ax=axes[1], cbar_kws={'label':'Participantes (n)'})
axes[1].set(title='Tamanho de cada segmento', xlabel='Café (xícaras/dia)', ylabel='Idade (anos)')
savefig('08_segmentos_idade')
display(analise.groupby(['Faixa_Idade','Stress_Level','Faixa_Cafe'], observed=True).Sleep_Hours.agg(n='size', media='mean').round(2))
''')
md('''### Principais Descobertas
Os contrastes abaixo são descritivos. Os limites de xícaras são apenas recortes analíticos, não recomendações médicas.
''')
code('''diferenca_cafe = resumo_cafe.loc['Até 2','media'] - resumo_cafe.loc['Mais de 4','media']
diferenca_estresse = resumo_estresse.loc['Low','media'] - resumo_estresse.loc['High','media']
descobertas = (
    f'- Participantes com mais de 4 xícaras/dia dormem em média **{resumo_cafe.loc["Mais de 4","media"]:.2f} h**, '
    f'contra **{resumo_cafe.loc["Até 2","media"]:.2f} h** no grupo de até 2 xícaras; contraste de **{diferenca_cafe:.2f} h**.\\n'
    f'- Estresse alto: **{resumo_estresse.loc["High","media"]:.2f} h** de sono; estresse baixo: '
    f'**{resumo_estresse.loc["Low","media"]:.2f} h**; contraste de **{diferenca_estresse:.2f} h**.\\n'
    f'- A diferença entre a maior e a menor média por gênero é **{resumo_genero.media.max()-resumo_genero.media.min():.2f} h**. '
    'Essa comparação isolada não controla diferenças de composição dos grupos.\\n'
    '- Café e cafeína estimada são redundantes; o modelo principal usará apenas o consumo em xícaras.\\n'
    '- A análise por idade exibe também o tamanho dos segmentos, evitando dar o mesmo peso a grupos de tamanhos distintos.\\n'
    '- A base é sintética; não há demonstração de efeito causal, representatividade populacional ou eficácia de intervenções.'
)
display(Markdown(descobertas))
''')
md('''## Entrega 3 — Modelo preditivo
### 6. Auditoria do alvo
`Sleep_Hours` é conhecida apenas quando o sono já foi observado e está fortemente ligada à própria definição do alvo nesta base.
O cenário principal tenta prever a qualidade com hábitos e características, sem utilizar a duração do sono nem condições de saúde simultâneas.
''')
code('''display(dados.groupby('Sleep_Quality').Sleep_Hours.agg(['min','max','mean','count']).reindex(ORDER).round(3))
display(pd.crosstab(pd.cut(dados.Sleep_Hours, [2.9,5,6,8,10], include_lowest=True), dados.Sleep_Quality).reindex(columns=ORDER))
display(Markdown('As faixas observadas são quase separadas por classe. Isso sugere que prever a qualidade usando '
                 '`Sleep_Hours` pode reconstruir a regra de geração do alvo, em vez de aprender uma previsão útil antes do sono.'))
display(pd.crosstab(dados.Stress_Level, dados.Sleep_Quality).reindex(columns=ORDER))
display(Markdown('**Outro sinal estrutural:** nesta base, todo estresse alto corresponde a `Poor`, todo estresse médio '
                 'a `Fair`, e o estresse baixo reúne apenas `Good` e `Excellent`. Não é possível confirmar a direção '
                 'de geração dessas variáveis. O modelo principal mantém estresse como hábito informado, mas sua '
                 'avaliação pode ser otimista. Por isso também medimos um cenário sem estresse e sem sua interação.'))
''')
md('''### 7. Preparação e divisão
A feature derivada `Coffee_Stress_Interaction` combina xícaras de café e um código ordinal de estresse (0, 1, 2).
Ela é uma hipótese exploratória, não uma escala validada. A transformação é aplicada dentro de cada pipeline.
Imputação, padronização e codificação são ajustadas somente no treino de cada dobra.
''')
code('''excluir = ['ID', 'Sleep_Quality', 'Caffeine_mg', 'Sleep_Hours', 'Health_Issues']
X = dados.drop(columns=excluir)
y = dados.Sleep_Quality
idx_treino, idx_teste = train_test_split(dados.index, test_size=.2, random_state=SEED, stratify=y)
X_train, X_test = X.loc[idx_treino], X.loc[idx_teste]
y_train, y_test = y.loc[idx_treino], y.loc[idx_teste]
assert set(idx_treino).isdisjoint(idx_teste)
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist() + ['Coffee_Stress_Interaction']
def make_pipeline(classifier, extra_num=None):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', numeric, num_cols + (extra_num or [])), ('cat', categorical, cat_cols)])
    return Pipeline([('features', HabitFeatures()), ('preprocess', pre), ('model', classifier)])
modelos = {
    'Referência majoritária': make_pipeline(DummyClassifier(strategy='most_frequent')),
    'Regressão logística': make_pipeline(LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED)),
    'Floresta aleatória': make_pipeline(RandomForestClassifier(n_estimators=180, max_depth=10, min_samples_leaf=8,
                         class_weight='balanced', random_state=SEED, n_jobs=-1))
}
display(pd.DataFrame({'treino': y_train.value_counts(), 'teste': y_test.value_counts()}).reindex(ORDER))
''')
md('''### 8. Seleção por validação cruzada no treino
Três dobras estratificadas. A configuração dos modelos é fixada previamente; não fazemos busca com o teste.
A dispersão entre dobras mostra a estabilidade interna, não um intervalo de confiança populacional.
''')
code('''cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
resultados_cv = []
for nome, pipeline in modelos.items():
    scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                            scoring={'f1_macro':'f1_macro','accuracy':'accuracy'}, return_train_score=True, n_jobs=1)
    resultados_cv.append({'modelo': nome, 'f1_macro_cv': scores['test_f1_macro'].mean(),
        'desvio_f1_cv': scores['test_f1_macro'].std(), 'acuracia_cv': scores['test_accuracy'].mean(),
        'f1_treino_cv': scores['train_f1_macro'].mean()})
cv_table = pd.DataFrame(resultados_cv).sort_values('f1_macro_cv', ascending=False)
elegiveis = cv_table[cv_table.modelo != 'Referência majoritária']
nome_melhor = elegiveis.iloc[0]['modelo']
display(cv_table.round(4))
print('Selecionado antes de abrir o teste:', nome_melhor)
cv_table.to_csv(ROOT / 'reports/comparacao_validacao.csv', index=False)
''')
md('''### 9. Avaliação no teste reservado
Os modelos são ajustados no treino completo; o teste permanece fora do pré-processamento e da seleção.
A comparação a seguir é descritiva e não será usada para trocar a escolha feita na validação cruzada.
''')
code('''resultados = []
previsoes = {}
for nome, pipeline in modelos.items():
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    previsoes[nome] = pred
    resultados.append({'modelo': nome, 'acuracia_treino': accuracy_score(y_train, pipeline.predict(X_train)),
                      'acuracia_teste': accuracy_score(y_test, pred),
                      'f1_macro_treino': f1_score(y_train, pipeline.predict(X_train), average='macro'),
                      'f1_macro_teste': f1_score(y_test, pred, average='macro')})
resultados = pd.DataFrame(resultados).merge(cv_table, on='modelo')
display(resultados.round(4))
resultados.to_csv(ROOT / 'reports/comparacao_modelos.csv', index=False)
melhor = modelos[nome_melhor]
pred = previsoes[nome_melhor]
relatorio = classification_report(y_test, pred, labels=ORDER, output_dict=True, zero_division=0)
display(pd.DataFrame(relatorio).T.round(3))
pd.DataFrame(relatorio).T.to_csv(ROOT / 'reports/classificacao_melhor_modelo.csv')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=ORDER, cmap='Blues', colorbar=False, ax=axes[0])
axes[0].set(title=f'Matriz de confusão — {nome_melhor}', xlabel='Classe prevista', ylabel='Classe real')
ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=ORDER, normalize='true', cmap='Blues', colorbar=False,
                                      values_format='.2f', ax=axes[1])
axes[1].set(title='Proporção dentro de cada classe real', xlabel='Classe prevista', ylabel='Classe real')
savefig('09_matrizes_confusao')
''')
code('''fig, ax = plt.subplots(figsize=(10,5))
plot = resultados.set_index('modelo')[['f1_macro_treino','f1_macro_cv','f1_macro_teste']]
plot.plot.barh(ax=ax)
ax.set(xlabel='F1 macro (0 a 1)', xlim=(0,1), ylabel='', title='Generalização: treino, validação e teste')
ax.legend(['Treino completo','Validação cruzada','Teste reservado'], loc='lower right')
savefig('10_comparacao_modelos')
linha = resultados.set_index('modelo').loc[nome_melhor]
gap = linha.f1_macro_treino - linha.f1_macro_cv
display(Markdown(f'O modelo selecionado foi **{nome_melhor}**. F1 macro de validação: **{linha.f1_macro_cv:.3f}**; '
                 f'F1 macro no teste: **{linha.f1_macro_teste:.3f}**; acurácia no teste: **{linha.acuracia_teste:.3f}**. '
                 f'A diferença treino–validação é **{gap:.3f}**. '
                 'Uma diferença positiva sugere sobreajuste, mas não há um limiar universal. '
                 'Desempenho baixo também pode refletir informação insuficiente nos hábitos, e não apenas underfitting.'))
''')
md('''### 10. Cenário diagnóstico com horas de sono
Este experimento usa a mesma divisão. Ele não substitui o modelo principal e não é uma previsão antes do sono.
Serve para mostrar o risco de reportar uma acurácia muito alta graças a uma variável quase definidora do alvo.
''')
code('''X_diag = dados[X.columns.tolist() + ['Sleep_Hours']]
diagnostico = make_pipeline(RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=4,
                             class_weight='balanced', random_state=SEED, n_jobs=-1), extra_num=['Sleep_Hours'])
diagnostico.fit(X_diag.loc[idx_treino], y_train)
pred_diag = diagnostico.predict(X_diag.loc[idx_teste])
metricas_diag = {'acuracia_teste': accuracy_score(y_test,pred_diag),
                 'f1_macro_teste': f1_score(y_test,pred_diag,average='macro')}
display(pd.DataFrame([{'cenario':'Principal, sem Sleep_Hours', 'acuracia_teste':linha.acuracia_teste,
                       'f1_macro_teste':linha.f1_macro_teste},
                      {'cenario':'Diagnóstico, com Sleep_Hours', **metricas_diag}]).round(4))
display(Markdown('Se o cenário diagnóstico superar muito o principal, isso confirma a sensibilidade à definição '
                 'do alvo e à disponibilidade temporal das variáveis; não prova utilidade em clientes reais.'))
''')
md('''### 11. Importância preditiva
A importância por permutação é calculada no treino para evitar novas decisões baseadas no teste.
Mede a perda de F1 macro ao embaralhar uma variável; não representa causalidade. A redundância e as interações podem redistribuir essas importâncias.
''')
code('''from sklearn.base import clone
# Sensibilidade: descarta tanto o estresse original quanto a feature derivada que o utiliza.
pre_sem_estresse = ColumnTransformer([
    ('num', clone(melhor.named_steps['preprocess'].named_transformers_['num']),
     [c for c in num_cols if c != 'Coffee_Stress_Interaction']),
    ('cat', clone(melhor.named_steps['preprocess'].named_transformers_['cat']),
     [c for c in cat_cols if c != 'Stress_Level'])])
sem_estresse = Pipeline([('features', HabitFeatures()), ('preprocess', pre_sem_estresse),
                        ('model', clone(melhor.named_steps['model']))])
sem_estresse.fit(X_train, y_train)
pred_sem_estresse = sem_estresse.predict(X_test)
metricas_sem_estresse = {'acuracia_teste': accuracy_score(y_test, pred_sem_estresse),
                        'f1_macro_teste': f1_score(y_test, pred_sem_estresse, average='macro')}
display(pd.DataFrame([{'cenario':'Principal: hábitos com estresse', 'f1_macro_teste':linha.f1_macro_teste,
                       'acuracia_teste':linha.acuracia_teste},
                      {'cenario':'Sensibilidade: sem estresse e sem interação', **metricas_sem_estresse}]).round(4))
display(Markdown(f'Sem estresse e sua interação, o F1 macro é **{metricas_sem_estresse["f1_macro_teste"]:.3f}**. '
                 'A diferença em relação ao cenário principal mede a dependência do modelo dessa estrutura sintética. '
                 'Não tratar o desempenho principal como evidência de uma previsão prospectiva validada.'))
''')
code('''imp = permutation_importance(melhor, X_train.sample(1200, random_state=SEED),
                              y_train.loc[X_train.sample(1200, random_state=SEED).index],
                              scoring='f1_macro', n_repeats=3, random_state=SEED, n_jobs=1)
importancias = pd.DataFrame({'variavel':X.columns, 'importancia_media':imp.importances_mean,
                            'desvio':imp.importances_std}).sort_values('importancia_media')
fig, ax = plt.subplots(figsize=(10,6))
ax.barh(importancias.variavel, importancias.importancia_media, xerr=importancias.desvio, color='#287D8E')
ax.axvline(0, color='black', linewidth=.8)
ax.set(title='Importância por permutação — treino, amostra de 1.200', xlabel='Redução do F1 macro (média ± desvio em 3 repetições)')
savefig('11_importancia')
importancias.to_csv(ROOT / 'reports/importancias.csv', index=False)
''')
md('''### 12. Exportação dos dados e do modelo
O modelo salvo mantém o ajuste feito apenas no treino, preservando o significado da avaliação no teste.
O pipeline inclui feature derivada, tratamento de nulos, codificação e classificador.
O CSV processado usa o pré-processador ajustado no treino e inclui alvo, ID de referência e indicação treino/teste.
ID e indicação de partição são metadados e não devem virar preditores.
''')
code('''X_features = melhor.named_steps['features'].transform(X)
matriz = melhor.named_steps['preprocess'].transform(X_features)
colunas = melhor.named_steps['preprocess'].get_feature_names_out()
processado = pd.DataFrame(matriz, columns=colunas, index=dados.index)
processado.insert(0, 'ID_referencia', dados.ID)
processado['Sleep_Quality'] = y
processado['particao'] = np.where(processado.index.isin(idx_treino), 'treino', 'teste')
processado.to_csv(ROOT / 'data/processed/dataset_processado.csv', index=False)
joblib.dump(melhor, ROOT / 'models/modelo_sono.joblib', compress=3)
recarregado = joblib.load(ROOT / 'models/modelo_sono.joblib')
assert np.array_equal(recarregado.predict(X_test.head(20)), pred[:20])
pd.DataFrame({'ID':dados.loc[idx_teste,'ID'], 'real':y_test, 'previsto':pred}).to_csv(ROOT / 'reports/previsoes_teste.csv', index=False)
metadata = {'modelo':nome_melhor, 'seed':SEED, 'target':'Sleep_Quality', 'classes':ORDER,
            'features_entrada':X.columns.tolist(), 'excluidas':excluir, 'n_treino':len(X_train), 'n_teste':len(X_test),
            'f1_macro_cv':float(linha.f1_macro_cv), 'f1_macro_teste':float(linha.f1_macro_teste),
            'acuracia_teste':float(linha.acuracia_teste), 'diagnostico_com_sleep_hours':metricas_diag,
            'sensibilidade_sem_estresse':metricas_sem_estresse,
            'fonte':proveniencia, 'python':platform.python_version(),
            'pacotes':{p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','matplotlib','seaborn','joblib']}}
(ROOT / 'models/metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
assert len(processado) == len(dados) and not processado.drop(columns=['Sleep_Quality','particao']).isna().any().any()
print('Modelo salvo e recarregado com previsões idênticas. Dataset exportado:', processado.shape)
''')
md('''## Recomendações para o Negócio
- Usar os resultados como exercício de segmentação e formulação de hipóteses, sem tratar a base sintética como evidência sobre clientes reais.
- Em um estudo real, registrar horário do café, outras fontes de cafeína, rotina de trabalho e duração/qualidade do sono em momentos definidos. Esses fatores não estão disponíveis aqui.
- Avaliar conjuntamente café e estresse. As diferenças observadas entre grupos não demonstram que reduzir café produzirá o mesmo ganho em horas de sono.
- Comparar o modelo com uma referência simples e acompanhar recall por classe; uma acurácia agregada pode esconder falhas em `Poor` e `Excellent`.
- Antes de qualquer uso operacional, validar prospectivamente com dados reais, consentimento e avaliação de representatividade, calibração e desempenho por grupo.
- Não utilizar o classificador para diagnóstico, prescrição ou decisões individuais de saúde.

## Limitações e próximos passos
Não há evidência do mecanismo de geração completo da base, sequência temporal dos hábitos, medição objetiva ou relação causal.
O alvo tem faixas fortemente relacionadas às horas de sono e categorias quase determinadas pelo estresse. O cenário principal também depende dessa estrutura; a sensibilidade sem estresse explicita essa limitação.
Não houve busca extensiva de hiperparâmetros: o foco é uma comparação reproduzível, com teste reservado e validação cruzada.
''')
code('''resumo = (f'# Resultados — café e sono\\n\\nBase sintética: {len(dados):,} registros e {len(dados.columns)} colunas.\\n\\n'
          f'## Principais Descobertas\\n{descobertas}\\n\\n'
          f'## Classificação\\nModelo selecionado por F1 macro na validação cruzada: **{nome_melhor}**.\\n'
          f'- F1 macro na validação: {linha.f1_macro_cv:.4f}.\\n'
          f'- F1 macro no teste: {linha.f1_macro_teste:.4f}.\\n'
          f'- Acurácia no teste: {linha.acuracia_teste:.4f}.\\n'
          f'- Cenário diagnóstico com Sleep_Hours: F1 macro {metricas_diag["f1_macro_teste"]:.4f}.\\n\\n'
          f'- Sensibilidade sem estresse e sem sua interação: F1 macro {metricas_sem_estresse["f1_macro_teste"]:.4f}.\\n\\n'
          'O estresse quase determina o alvo nesta base: alto corresponde a Poor; médio a Fair; baixo a Good ou Excellent. '
          'A dependência dessa estrutura limita a interpretação do desempenho principal.\\n\\n'
          'Não inferir causalidade, diagnóstico ou validade em uma população real a partir destes dados sintéticos.\\n')
(ROOT / 'reports/resultados.md').write_text(resumo, encoding='utf-8')
display(Markdown(resumo))
print('Verificações finais: partições sem sobreposição, alvo presente, exportação sem nulos e modelo recarregável.')
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
                                                 'language_info': {'name':'python','version':'3.12'}})
nbf.validate(nb)
nbf.write(nb, ROOT / 'notebooks/01_cafe_sono.ipynb')
print('Notebook criado:', len(cells), 'células')
