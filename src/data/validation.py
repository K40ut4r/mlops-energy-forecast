"""
Validation des données avec Great Expectations (version simplifiée).
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validateur de données basique (sans GE installé, version manuelle)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors = []

    def validate_schema(self, expected_columns: list[str]) -> bool:
        """Vérifie que toutes les colonnes attendues sont présentes."""
        missing = set(expected_columns) - set(self.df.columns)
        if missing:
            self.errors.append(f"Colonnes manquantes: {missing}")
            return False
        return True

    def validate_types(self, expected_types: dict[str, str]) -> bool:
        """Vérifie les types des colonnes."""
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual = str(self.df[col].dtype)
                if expected_type not in actual:
                    self.errors.append(
                        f"Type incorrect pour {col}: {actual} (attendu: {expected_type})"
                    )
        return len(self.errors) == 0

    def validate_no_nulls(self, columns: list[str]) -> bool:
        """Vérifie qu'il n'y a pas de valeurs nulles."""
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                self.errors.append(f"Valeurs nulles dans {col}")
        return len(self.errors) == 0

    def validate_range(self, column: str, min_val: float, max_val: float) -> bool:
        """Vérifie qu'une colonne est dans un range."""
        if column in self.df.columns:
            if (self.df[column] < min_val).any() or (self.df[column] > max_val).any():
                self.errors.append(f"{column} hors range [{min_val}, {max_val}]")
        return len(self.errors) == 0

    def run_all(self) -> dict:
        """Exécute toutes les validations."""
        # Validation du dataset brut
        self.validate_schema(
            [
                "Global_active_power",
                "Global_reactive_power",
                "Voltage",
                "Global_intensity",
                "Sub_metering_1",
                "Sub_metering_2",
                "Sub_metering_3",
            ]
        )

        self.validate_no_nulls(["Global_active_power", "Voltage"])

        self.validate_range("Global_active_power", 0, 20)
        self.validate_range("Voltage", 200, 260)

        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "timestamp": pd.Timestamp.now().isoformat(),
        }


def main():
    """Point d'entrée pour validation."""
    logging.basicConfig(level=logging.INFO)

    df = pd.read_parquet("data/processed/cleaned_data.parquet")
    validator = DataValidator(df)
    results = validator.run_all()

    # Sauvegarde
    output_path = Path("metrics/validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Validation: {'✅ PASS' if results['valid'] else '❌ FAIL'}")
    if results["errors"]:
        for err in results["errors"]:
            logger.error(f"  - {err}")


if __name__ == "__main__":
    main()
