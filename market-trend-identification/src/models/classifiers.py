from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, StackingClassifier

class TrendClassifier:
    def __init__(self):
        self.classifiers = {
            'decision_tree': DecisionTreeClassifier(
                max_depth=10, 
                min_samples_split=5,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'ann': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        }
        
    def create_ensemble(self):
        """
        Create ensemble models
        """
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.classifiers['random_forest']),
                ('xgb', self.classifiers['xgboost']),
                ('svm', self.classifiers['svm'])
            ],
            voting='soft'
        )
        
        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', self.classifiers['random_forest']),
                ('xgb', self.classifiers['xgboost']),
                ('knn', self.classifiers['knn'])
            ],
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        self.classifiers['voting'] = voting_clf
        self.classifiers['stacking'] = stacking_clf
        
        return voting_clf, stacking_clf
