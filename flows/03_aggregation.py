"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Paso 28: estadísticas descriptivas por grupos."""
    return {
        "age_by_survival": df.groupby("Survived")["Age"].describe().reset_index(),
        "survival_by_sex": df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index(),
        "survival_by_pclass": (
            df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"]).reset_index()
        ),
        "survival_by_alone": (
            df.groupby("Alone")["Survived"].agg(["mean", "sum", "count"]).reset_index()
        ),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Paso 29: devuelve {'chi2': float, 'p_value': float, 'dof': int}."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, gl, _esperados = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(gl)}


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34: tasa de supervivencia con Pclass en filas y Sex en columnas."""
    tabla = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
    return tabla.reset_index().rename(columns=str)


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (1): tasa y número de pasajeros por puerto de embarque."""
    return df.groupby("Embarked")["Survived"].agg(["mean", "count"]).reset_index()


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (2): tasa de supervivencia por puerto, estratificada por clase."""
    tabla = df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    return tabla.reset_index().rename(columns=str)


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (1): agrega 'FareQ', cuartil de Fare calculado dentro de cada Pclass."""
    df = df.copy()
    df["FareQ"] = df.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return df


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (2): tasa de supervivencia por cuartil de tarifa (filas) y clase (columnas)."""
    df = add_fare_quartile(df)
    tabla = df.pivot_table(values="Survived", index="FareQ", columns="Pclass", aggfunc="mean")
    return tabla.reset_index().rename(columns=str)


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37: tasa de supervivencia y cantidad de pasajeros por tamaño de familia."""
    return df.groupby("Familia")["Survived"].agg(["mean", "count"]).reset_index()


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Paso 38 (1): máscara booleana de atípicos según la regla del IQR."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38 (2-3): cantidad de atípicos en Fare, Age (original) y Familia.

    Para Fare añade la tasa de supervivencia de los atípicos frente al resto.
    """
    # Edad original (sin imputar); si la tabla no la trae, se usa Age
    edad = df["AgeOriginal"] if "AgeOriginal" in df.columns else df["Age"]
    filas = []
    for nombre, serie in [("Fare", df["Fare"]), ("Age", edad), ("Familia", df["Familia"])]:
        mascara = outlier_mask_iqr(serie.dropna()).reindex(serie.index, fill_value=False)
        fila = {
            "variable": nombre,
            "n_atipicos": int(mascara.sum()),
            "survival_atipicos": float("nan"),
            "survival_resto": float("nan"),
        }
        if nombre == "Fare":
            fila["survival_atipicos"] = float(df.loc[mascara, "Survived"].mean())
            fila["survival_resto"] = float(df.loc[~mascara, "Survived"].mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
