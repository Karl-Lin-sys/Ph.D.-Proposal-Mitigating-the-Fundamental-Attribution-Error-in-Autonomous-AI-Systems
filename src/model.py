import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class FAE_Mitigating_Model:
    """
    A context-aware model designed to mitigate the Fundamental Attribution Error (FAE).
    It adjusts for situational confounders using Inverse Probability Weighting (IPW).
    """
    def __init__(self):
        self.propensity_model = LogisticRegression(random_state=42)
        self.outcome_model = RandomForestClassifier(random_state=42, max_depth=5)
        self.naive_model = RandomForestClassifier(random_state=42, max_depth=5)
        
    def fit(self, X, context, y):
        """
        X: Features representing behavior (e.g., facial cues, actions).
        context: Situational context confounder.
        y: True underlying state/intent (often misattributed in FAE).
        """
        # 1. Train Naive Model (Prone to FAE)
        # The naive model only looks at behavior, ignoring the context completely.
        self.naive_model.fit(X, y)
        
        # 2. Train Causal Debiasing Model (Mitigates FAE)
        # Calculate propensity scores based on context to adjust weights
        self.propensity_model.fit(context, y)
        probs = self.propensity_model.predict_proba(context)
        
        # IPW: Weight inversely by the probability of the treatment (or in this case, label conditioned on context)
        # to break the confounding association between context and behavior.
        # Ensure numerical stability
        eps = 1e-6
        probs = np.clip(probs, eps, 1 - eps)
        
        # Weight generation: If y=1, weight = 1/P(y=1|C). If y=0, weight = 1/P(y=0|C)
        weights = np.zeros(len(y))
        for i in range(len(y)):
            label_idx = list(self.propensity_model.classes_).index(y[i])
            weights[i] = 1.0 / probs[i, label_idx]
            
        # Normalize weights
        weights = weights / np.mean(weights)
        
        # Fit the outcome model with context included and IPW weights
        X_context = np.hstack([X, context])
        self.outcome_model.fit(X_context, y, sample_weight=weights)
        
    def predict_naive(self, X):
        return self.naive_model.predict(X)
        
    def predict_debiasing(self, X, context):
        X_context = np.hstack([X, context])
        return self.outcome_model.predict(X_context)

    def evaluate_fairness(self, X_test, context_test, y_test):
        """
        Evaluates how heavily the models rely on the context vs. actual behavior.
        Models with high FAE will have high error rates when the context differs 
        from typical expectations (out-of-distribution).
        """
        naive_acc = self.naive_model.score(X_test, y_test)
        
        X_context_test = np.hstack([X_test, context_test])
        debiased_acc = self.outcome_model.score(X_context_test, y_test)
        
        return {
            'naive_accuracy': naive_acc,
            'debiased_accuracy': debiased_acc
        }
