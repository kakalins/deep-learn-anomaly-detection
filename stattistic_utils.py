import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, RocCurveDisplay

class Statistics:

    def get_confusion_matrix(y_truth, y_pred):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i, value in enumerate(y_truth):
            if value == y_pred[i]:
                if value == 1:
                    tn += 1     # True negatives
                else:
                    tp += 1     # True positives
            else:
                if value == 1:
                    fn += 1
                else:
                    fp += 1
        return (tp, fp, tn, fn)
    # Receives the y_true (ground truth values) and y_pred (model predicted values) and returns accuracy, balanced accuracy, specificity, precision and balanced accuracy
    def calc_metrics(self,y_true, y_pred):
        accuracy = 0.
        sensibility = 0.
        precision = 0.
        specificity = 0.
        balanced_accuracy = 0.

        confusion_m = confusion_matrix(y_true, y_pred)
        print(confusion_m)
    
        (tn, fp, fn, tp) = confusion_m.ravel()
        #(tp, fp, tn, fn) = self.get_confusion_matrix(y_true, y_pred)
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
        sensibility = float(tp) / (tp + fn)
        precision = float(tp) / (tp + fp)
        specificity = float(tn) / (tn + fp)

        if (sensibility == 0 or precision == 0):
            f1_score = 0.
        else:
            f1_score = 2 * (sensibility * precision) / (sensibility + precision)
        balanced_accuracy = (sensibility + specificity) / 2.
        metrics = {
            'accuracy': accuracy, 
            'sensibility': sensibility, 
            'specificity': specificity,
            'precision': precision, 
            'balanced_accuracy': balanced_accuracy, 
            'f1-score': f1_score
        }
        return metrics
    
    # Receives a dictonaire of metrics and print it - receives metrics and returns void
    def print_metrics(self, metrics):
        print(f"Accuracy: {round(metrics['accuracy'] * 100, 3)}%")
        print(f"Sensitivity: {round(metrics['sensibility'] * 100, 3)}%")
        print(f"Specificity: {round(metrics['specificity'] * 100, 3)}%")
        print(f"Precision/Recall: {round(metrics['precision'] * 100, 3)}%")
        print(f"Balanced Accuracy: {round(metrics['balanced_accuracy'] * 100, 3)}%")
        print(f"F1_Score: {round(metrics['f1-score'] * 100, 3)}%")

    # Show a confusion matrix printed on prompt - receives y_true and y_predicted
    def get_confusion_matrix(self, y_true, y_pred):
        print('--'*10 + 'CONFUSION MATRIX' + '--'*10)
        print(confusion_matrix(y_true, y_pred))

    def show_roc_curve(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        auc = roc_auc_score(y_true, y_pred)
        RocCurveDisplay.from_predictions(y_true, y_pred, pos_label=1)
        return (fpr, tpr, thresholds, auc)
    
    def save_metrics(self, metrics, path, name):
        df = pd.DataFrame([metrics])
        df.to_csv(path + name + ".csv")


    