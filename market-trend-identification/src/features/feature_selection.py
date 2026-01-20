from sklearn.feature_selection import SelectKBest, RFE, f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class FeatureSelector:
    def __init__(self, n_features=50):
        self.n_features = n_features
        self.selector = None
        
    def select_features(self, X, y, method='random_forest'):
        """
        Select most important features using various methods
        """
        if method == 'kbest':
            # SelectKBest with ANOVA F-value
            selector = SelectKBest(score_func=f_classif, k=self.n_features)
            X_selected = selector.fit_transform(X, y)
            self.selector = selector
            return X_selected, selector.get_support()
            
        elif method == 'random_forest':
            # Using Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Select top n_features
            selected_indices = indices[:self.n_features]
            X_selected = X[:, selected_indices]
            
            # Create mask for selected features
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[selected_indices] = True
            
            self.selected_indices = selected_indices
            return X_selected, mask
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=rf, n_features_to_select=self.n_features)
            X_selected = rfe.fit_transform(X, y)
            self.selector = rfe
            return X_selected, rfe.support_
