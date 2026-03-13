import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle


def train_workforce_model():
    # 1. Load the dataset
    try:
        df = pd.read_csv('workforce_data.csv')
    except FileNotFoundError:
        print("❌ Error: 'workforce_data.csv' not found. Run the generator script first.")
        return

    # 2. Prepare Features (X) and Target (y)
    # We drop 'Name' because names don't predict risk
    X = df.drop(columns=['Name', 'Risk_Level'])
    y = df['Risk_Level']

    # 3. Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and Train the Random Forest Classifier
    # n_estimators=100 means we are using a "forest" of 100 decision trees
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"✅ Model Training Complete!")
    print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, predictions))

    # 6. Save the model to a pickle file
    with open('workforce_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("💾 Model saved as 'workforce_model.pkl'")


if __name__ == "__main__":
    train_workforce_model()