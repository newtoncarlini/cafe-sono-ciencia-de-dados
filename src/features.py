"""Transformações reutilizáveis, sem ajuste a partir da variável alvo."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HabitFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dados = X.copy()
        # Interação exploratória: não representa um limiar clínico.
        dados['Coffee_Stress_Interaction'] = (
            dados['Coffee_Intake'] * dados['Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
        )
        return dados
