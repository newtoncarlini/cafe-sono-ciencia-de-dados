"""Atualiza somente apresentação após a execução; não altera cálculos ou previsões."""
from pathlib import Path
import json
import nbformat

root = Path(__file__).resolve().parent
metadata = json.loads((root / 'models/metadata.json').read_text(encoding='utf-8'))
path = root / 'notebooks/01_cafe_sono.ipynb'
nb = nbformat.read(path, as_version=4)
inicio = nb.cells[0].source
old = ('O resumo com resultados é calculado ao final da execução e também salvo em `reports/resultados.md`.\n'
       'Este notebook contém as três entregas, as verificações de qualidade e os arquivos para reprodução.')
novo = (f'- **10.000 registros**, 16 variáveis; sem nulos após preservar corretamente a categoria textual `None`.\n'
        f'- Mais de 4 xícaras/dia: **6,28 h** de sono, contra **6,88 h** no grupo de até 2 xícaras. Associação descritiva, não causal.\n'
        f'- Modelo escolhido na validação: **{metadata["modelo"]}**. Teste reservado: '
        f'**{metadata["acuracia_teste"]:.2%} de acurácia** e **{metadata["f1_macro_teste"]:.3f} de F1 macro**.\n'
        f'- Sem estresse e sua interação, o F1 macro cai para **{metadata["sensibilidade_sem_estresse"]["f1_macro_teste"]:.3f}**. '
        'O desempenho depende fortemente da estrutura sintética do alvo.\n'
        '- As três entregas, recomendações, limitações e exportações estão documentadas abaixo.')
nb.cells[0].source = inicio.replace(old, novo)
for cell in nb.cells:
    if cell.cell_type == 'code':
        for output in cell.outputs:
            if 'image/png' in output.get('data', {}):
                output.metadata.setdefault('image/png', {})['alt'] = 'Gráfico da análise de café e sono; título, eixos e interpretação na própria seção.'
nbformat.validate(nb)
nbformat.write(nb, path)
