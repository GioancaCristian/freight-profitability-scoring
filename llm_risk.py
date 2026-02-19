import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@dataclass
class RiskOutput:
    risk_probability: float
    adjustment: int
    rationale: str


class RiskModel:

    def __init__(self):
    #Initialize RandomForestClassifier.
        self.model = RandomForestClassifier(
            n_estimators=100,   # number of trees
            max_depth=6,        # limit tree depth (prevents overfitting)
            random_state=42
        )

        self.features = [
            "distance_km",
            "deadhead_km",
            "border_crossings",
            "waiting_hours",
            "payment_terms_days",
            "margin_pct"
        ]

        self.trained = False

    def prepare_target(self, df):
        """
        Create binary target:
        1 if transport makes loss
        0 if transport is profitable
        """
        df["loss_flag"] = np.where(df["profit_eur"] < 0, 1, 0)
        return df

    def train(self, df):
        #Train classifier to predict probability of loss.
        df = self.prepare_target(df)

        X = df[self.features]
        y = df["loss_flag"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        self.trained = True

    def predict_risk(self, row):
        #Predict probability of financial loss.

        if not self.trained:
            raise Exception("Model not trained")

        X = pd.DataFrame([row])[self.features]

        # Get probability of class 1 (loss)
        probabilities = self.model.predict_proba(X)[0]
        risk_probability = float(probabilities[1])  # class 1 = loss

        # Map probability to score adjustment (-10 to 0)
        adjustment = int(-10 * risk_probability)

        rationale = (
            f"Predicted loss probability {round(risk_probability, 2)} "
            f"based on operational and financial features."
        )

        return RiskOutput(
            risk_probability=risk_probability,
            adjustment=adjustment,
            rationale=rationale
        )
