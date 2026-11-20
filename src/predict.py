"""Inferência com o pipeline exportado; executar da raiz com python -m src.predict."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV com as colunas originais de entrada')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / 'models/metadata.json').read_text(encoding='utf-8'))
    dados = pd.read_csv(args.input, keep_default_na=False, na_values=['', 'NA', 'N/A', 'null'])
    faltantes = set(metadata['features_entrada']) - set(dados.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes: {sorted(faltantes)}')
    modelo = joblib.load(root / 'models/modelo_sono.joblib')
    resultado = pd.DataFrame({'Sleep_Quality_prevista': modelo.predict(dados[metadata['features_entrada']])})
    if 'ID' in dados:
        resultado.insert(0, 'ID', dados.ID)
    destino = Path(args.output)
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(destino, index=False)
    print(f'{len(resultado)} previsões salvas em {destino}')


if __name__ == '__main__':
    main()
