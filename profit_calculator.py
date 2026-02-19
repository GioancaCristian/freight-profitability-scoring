import pandas as pd

from llm_risk import RiskModel

def calculate_profit(row):
    if row["rate_type"] == "per_km":
        revenue = row["rate_eur_per_km"] * row["distance_km"]
    else:
        revenue = row["total_contract_value"]

    revenue_after_broker = revenue * (1 - row["broker_fee_pct"] / 100)

    total_km = row["distance_km"] + row["deadhead_km"]

    fuel_cost = (
        (total_km / 100)
        * row["fuel_l_per_100km"]
        * row["fuel_price_eur_per_l"]
    )

    maintenance_cost = total_km * row["maintenance_cost_per_km"]

    insurance_cost = total_km * row["insurance_cost_per_km"]

    driver_salary_cost = (
        row["estimated_transit_time_hours"]
        * row["driver_cost_per_hour"]
    )

    driver_total_cost = driver_salary_cost + row["driver_per_diem"]

    waiting_cost = (
        row["waiting_hours"]
        * row["driver_cost_per_hour"]
    )

    total_cost = (
        fuel_cost
        + maintenance_cost
        + insurance_cost
        + driver_total_cost
        + waiting_cost
        + row["toll_cost_eur"]
    )

    profit = round(revenue_after_broker - total_cost,2)


    if revenue_after_broker > 0:
        margin_pct = (profit / revenue_after_broker) * 100
    else:
        margin_pct = 0

    return pd.Series([revenue_after_broker, total_cost, profit, margin_pct])

def calculate_score(row):

    total_km = row["distance_km"] + row["deadhead_km"]

    margin = max(row["margin_pct"], 0)
    margin_capped = min(margin, 40)
    margin_score = (margin_capped / 40) * 60

    if total_km > 0:
        profit_per_km = row["profit_eur"] / total_km
    else:
        profit_per_km = 0

    profit_km_capped = min(max(profit_per_km, 0), 0.8)
    profit_km_score = (profit_km_capped / 0.8) * 15

    if total_km > 0:
        deadhead_ratio = row["deadhead_km"] / total_km
    else:
        deadhead_ratio = 1

    deadhead_capped = min(deadhead_ratio, 0.25)
    deadhead_score = (1 - deadhead_capped / 0.25) * 10

    if row["distance_km"] > 0:
        waiting_ratio = row["waiting_hours"] / (row["distance_km"] / 1000)
    else:
        waiting_ratio = 5

    waiting_capped = min(waiting_ratio, 5)
    waiting_score = (1 - waiting_capped / 5) * 10

    payment_days = row["payment_terms_days"]
    payment_capped = min(payment_days, 60)
    payment_score = (1 - (payment_capped - 15) / 45) * 5 if payment_days > 15 else 5
    payment_score = max(payment_score, 0)

    total_score = (
        margin_score
        + profit_km_score
        + deadhead_score
        + waiting_score
        + payment_score
    )

    return round(total_score, 2)

def main():
    df = pd.read_csv("Data/freight_profitability_dataset.csv")

    df[[
        "revenue_after_broker",
        "total_cost",
        "profit_eur",
        "margin_pct"
    ]] = df.apply(calculate_profit, axis=1)
    df.to_csv("Data/freight_profitability_with_profit.csv", index=False)

    df["profitability_score"] = df.apply(calculate_score, axis=1)
    df.to_csv("Data/freight_profitability_scored.csv", index=False)

    risk_model = RiskModel()
    risk_model.train(df)

    risk_probabilities = []
    adjustments = []
    rationales = []
    final_scores = []

    for _, row in df.iterrows():
        risk_output = risk_model.predict_risk(row.to_dict())

        risk_probabilities.append(risk_output.risk_probability)
        adjustments.append(risk_output.adjustment)
        rationales.append(risk_output.rationale)

        final_score = row["profitability_score"] + risk_output.adjustment
        final_score = round(max(0, min(100, final_score)),2)
        final_scores.append(final_score)

    df["risk_probability"] = risk_probabilities
    df["risk_adjustment"] = adjustments
    df["risk_rationale"] = rationales
    df["final_score"] = final_scores
    df.to_csv("Data/freight_profitability_final.csv", index=False)


    print(df[[
        "id",
        "revenue_after_broker",
        "total_cost",
        "profit_eur",
        "margin_pct",
        "profitability_score"
    ]].head())


if __name__ == "__main__":
    main()
