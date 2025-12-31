from src.preprocess import preprocess_data
from src.train_model import train_models
from src.evaluate import evaluate_model
import joblib
import os

def main():
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        "data/raw/heart.csv"
    )

    # Train models
    models = train_models(X_train, y_train)

    best_model = None
    best_score = 0

    # Evaluate models
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        print(f"\n{name} Performance:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        if metrics["Recall"] > best_score:
            best_score = metrics["Recall"]
            best_model = model

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    print("\nBest model saved successfully!")

if __name__ == "__main__":
    main()
