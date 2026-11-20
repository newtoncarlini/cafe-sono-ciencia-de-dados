"""Verifica as entregas após a execução completa do notebook."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import nbformat
import joblib

ROOT = Path(__file__).resolve().parent


def main():
    notebook = nbformat.read(ROOT / 'notebooks/01_cafe_sono.ipynb', as_version=4)
    nbformat.validate(notebook)
    codes = [c for c in notebook.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type == 'error' for c in codes for o in c.outputs)
    original = pd.read_csv(ROOT / 'data/raw/synthetic_coffee_health_10000.csv', keep_default_na=False)
    processed = pd.read_csv(ROOT / 'data/processed/dataset_processado.csv')
    predictions = pd.read_csv(ROOT / 'reports/previsoes_teste.csv')
    metadata = json.loads((ROOT / 'models/metadata.json').read_text(encoding='utf-8'))
    assert len(original) == len(processed) == 10000
    assert processed.ID_referencia.is_unique and processed.notna().all().all()
    assert processed.particao.value_counts().to_dict() == {'treino':8000, 'teste':2000}
    assert set(predictions.ID) == set(processed.loc[processed.particao == 'teste', 'ID_referencia'])
    assert not any('Sleep_Hours' in c or 'Caffeine_mg' in c or 'Health_Issues' in c for c in processed.columns)
    assert hashlib.sha256((ROOT / 'data/raw/synthetic_coffee_health_10000.csv').read_bytes()).hexdigest() == metadata['fonte']['sha256']
    # Recomposição independente de acurácia e F1 por contagens.
    accuracy = float((predictions.real == predictions.previsto).mean())
    fs = []
    for label in metadata['classes']:
        tp = ((predictions.real == label) & (predictions.previsto == label)).sum()
        fp = ((predictions.real != label) & (predictions.previsto == label)).sum()
        fn = ((predictions.real == label) & (predictions.previsto != label)).sum()
        fs.append(2*tp/(2*tp+fp+fn))
    assert np.isclose(accuracy, metadata['acuracia_teste'])
    assert np.isclose(np.mean(fs), metadata['f1_macro_teste'])
    model = joblib.load(ROOT / 'models/modelo_sono.joblib')
    test = original.set_index('ID').loc[predictions.ID, metadata['features_entrada']]
    assert np.array_equal(model.predict(test), predictions.previsto.to_numpy())
    assert len(list((ROOT / 'reports/figures').glob('*.png'))) == 11
    status = {'status':'aprovado', 'celulas_codigo_executadas':len(codes), 'figuras':11,
              'registros':10000, 'teste':2000, 'modelo_recarregado':True, 'metricas_recalculadas':True}
    (ROOT / 'reports/validacao.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
