from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, auc)
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_names):
        self.results = {}
        self.model_names = model_names
        
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return self.results[model_name]
    
    def generate_report(self):
        """
        Generate comprehensive evaluation report
        """
        report_data = []
        
        for model_name in self.model_names:
            if model_name in self.results:
                metrics = self.results[model_name]
                report_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}"
                })
                
        report_df = pd.DataFrame(report_data)
        return report_df
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_test, class_names):
        """
        Plot ROC curves for multi-class classification
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            if metrics['probabilities'] is not None:
                # One-vs-Rest ROC curve
                fpr = {}
                tpr = {}
                roc_auc = {}
                
                for i in range(len(class_names)):
                    fpr[i], tpr[i], _ = roc_curve(y_test == i, 
                                                 metrics['probabilities'][:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Plot micro-average ROC
                fpr["micro"], tpr["micro"], _ = roc_curve(
                    y_test.ravel(), metrics['probabilities'].ravel()
                )
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                plt.plot(fpr["micro"], tpr["micro"],
                        label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})',
                        lw=2)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Micro Average')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
