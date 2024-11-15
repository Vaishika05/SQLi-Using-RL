import pandas as pd
from joblib import load


def predict_sqli():
    # Load the saved model
    model = load("sqli_model.pkl")

    # Load the CSV files containing the generated queries
    simpleAgent_data = pd.read_csv("generated_queries_simpleAgent.csv")
    dqnAgent_data = pd.read_csv("generated_queries_dqnAgent.csv")

    # Preprocess the queries if necessary (e.g., handle NaN values, convert to strings)
    simpleAgent_data["Generated Queries"] = (
        simpleAgent_data["Generated Queries"].fillna("").astype(str)
    )
    dqnAgent_data["Generated Queries"] = (
        dqnAgent_data["Generated Queries"].fillna("").astype(str)
    )

    # Make predictions for each query in both CSV files
    simpleAgent_predictions = model.predict(simpleAgent_data["Generated Queries"])
    dqnAgent_predictions = model.predict(dqnAgent_data["Generated Queries"])

    # Add the predictions as new columns in both dataframes
    simpleAgent_data["Prediction"] = simpleAgent_predictions
    dqnAgent_data["Prediction"] = dqnAgent_predictions

    # Save the DataFrames with queries and predictions to new CSV files
    simpleAgent_data.to_csv("simpleAgent_with_predictions.csv", index=False)
    dqnAgent_data.to_csv("dqnAgent_with_predictions.csv", index=False)

    print("Predictions for Simple Agent and DQN Agent saved to respective CSV files.")


# Run the prediction function
if __name__ == "__main__":
    predict_sqli()
