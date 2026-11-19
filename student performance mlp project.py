from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_mlp_model.pkl"
BINARY_COLUMNS = [
    "schoolsup", "famsup", "paid",
    "activities", "higher", "internet"
]

def ask_number(label, minimum, maximum=None):
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = float(raw)
            if value < minimum or (maximum is not None and value > maximum):
                limit = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
                print(f"Please enter a value {limit}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")

def ask_yes_no(label):
    while True:
        value = input(f"{label} (yes/no): ").strip().lower()
        if value in ("yes", "no"):
            return 1 if value == "yes" else 0
        print("Please type yes or no.")

def main():
    if not MODEL_PATH.exists():
        print("Model file not found. Run 'python train.py' first.")
        return

    saved = joblib.load(MODEL_PATH)
    model, scaler, features = saved["model"], saved["scaler"], saved["features"]

    print("\n=== STUDENT PERFORMANCE PREDICTION ===")
    print("Enter the student's details below.\n")
    values = {
        "age": ask_number("Age", 15, 25),
        "studytime": ask_number("Weekly study time category (1-4)", 1, 4),
        "failures": ask_number("Number of past class failures (0-4)", 0, 4),
        "absences": ask_number("Number of school absences (0-93)", 0, 93),
    }
    for col in BINARY_COLUMNS:
        values[col] = ask_yes_no(col)

    student = pd.DataFrame([[values[col] for col in features]], columns=features)
    scaled = scaler.transform(student)
    prediction = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]

    print("\n----------- PREDICTION -----------")
    print("Predicted result:", "PASS" if prediction == 1 else "FAIL")
    print(f"Estimated pass probability: {probabilities[1] * 100:.2f}%")
    print(f"Estimated fail probability: {probabilities[0] * 100:.2f}%")
    print("----------------------------------")
    print("Educational demonstration only; this is not a guaranteed academic outcome.")

if __name__ == "__main__":
    main()
