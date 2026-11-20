# Resultados — café e sono

Base sintética: 10,000 registros e 16 colunas.

## Principais Descobertas
- Participantes com mais de 4 xícaras/dia dormem em média **6.28 h**, contra **6.88 h** no grupo de até 2 xícaras; contraste de **0.60 h**.
- Estresse alto: **4.45 h** de sono; estresse baixo: **7.25 h**; contraste de **2.80 h**.
- A diferença entre a maior e a menor média por gênero é **0.04 h**. Essa comparação isolada não controla diferenças de composição dos grupos.
- Café e cafeína estimada são redundantes; o modelo principal usará apenas o consumo em xícaras.
- A análise por idade exibe também o tamanho dos segmentos, evitando dar o mesmo peso a grupos de tamanhos distintos.
- A base é sintética; não há demonstração de efeito causal, representatividade populacional ou eficácia de intervenções.

## Classificação
Modelo selecionado por F1 macro na validação cruzada: **Floresta aleatória**.
- F1 macro na validação: 0.7679.
- F1 macro no teste: 0.7676.
- Acurácia no teste: 0.7635.
- Cenário diagnóstico com Sleep_Hours: F1 macro 0.9861.

- Sensibilidade sem estresse e sem sua interação: F1 macro 0.2476.

O estresse quase determina o alvo nesta base: alto corresponde a Poor; médio a Fair; baixo a Good ou Excellent. A dependência dessa estrutura limita a interpretação do desempenho principal.

Não inferir causalidade, diagnóstico ou validade em uma população real a partir destes dados sintéticos.
