import numpy as np
from model import FAE_Mitigating_Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic data that simulates a Fundamental Attribution Error scenario.
    Context (C): e.g., Environmental difficulty (0=Easy, 1=Hard).
    Internal Trait/State (Y): e.g., Competence or Intent (0=Low, 1=High).
    Behavior (X): The observed action/performance.
    
    The FAE occurs because observers (or naive models) look only at X to predict Y,
    ignoring that C heavily influences X.
    """
    np.random.seed(42)
    
    # Context (Situational factor)
    C = np.random.binomial(1, 0.5, n_samples).reshape(-1, 1)
    
    # True Trait (Y)
    Y = np.random.binomial(1, 0.5, n_samples)
    
    # Behavior (X) is heavily confounded by Context C.
    # If Context is Hard (C=1), Behavior is negatively impacted regardless of Trait.
    # If Context is Easy (C=0), Behavior is positively impacted.
    # Trait (Y) also impacts Behavior.
    X_mean = 2.0 * Y - 3.0 * C[:, 0] + np.random.normal(0, 1, n_samples)
    X = X_mean.reshape(-1, 1)
    
    return X, C, Y

def main():
    print("Generating Synthetic Data with Situational Confounders...")
    X, C, Y = generate_synthetic_data(2000)
    
    # Split data (simulating RCT / hold-out set)
    X_train, X_test, C_train, C_test, Y_train, Y_test = train_test_split(
        X, C, Y, test_size=0.2, random_state=42
    )
    
    # To truly show FAE, we evaluate on a "flipped" out-of-distribution context test set 
    # where the correlation between C and Y is artificially broken or modified,
    # or just standard test set where Naive model mistakes C's effect for Y's.
    
    model = FAE_Mitigating_Model()
    
    print("Training Naive AI Model (Prone to Fundamental Attribution Error)...")
    print("Training Causal Context-Aware Model (Debiased based on Systematic Reviews/RCTs)...")
    model.fit(X_train, C_train, Y_train)
    
    results = model.evaluate_fairness(X_test, C_test, Y_test)
    
    print("\n--- Evaluation Results ---")
    print(f"Naive Model Accuracy: {results['naive_accuracy']*100:.2f}%")
    print(f"Debiased Model Accuracy: {results['debiased_accuracy']*100:.2f}%")
    
    print("\nNaive Model Detailed Report:")
    y_pred_naive = model.predict_naive(X_test)
    print(classification_report(Y_test, y_pred_naive))
    
    print("\nCausal Context-Aware Model Detailed Report:")
    y_pred_debiased = model.predict_debiasing(X_test, C_test)
    print(classification_report(Y_test, y_pred_debiased))

if __name__ == "__main__":
    main()
