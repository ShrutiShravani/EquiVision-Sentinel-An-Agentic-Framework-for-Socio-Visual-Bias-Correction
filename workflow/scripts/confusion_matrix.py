import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import mlflow

def log_confusion_matrix(all_targets, all_outputs, class_names, fold, epoch):
    # Convert probabilities to binary 0/1
    preds_binary = (all_outputs > 0.5).astype(int)
    mcm = multilabel_confusion_matrix(all_targets, preds_binary)
    
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(len(class_names)):
        sns.heatmap(mcm[i], annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'Class: {class_names[i]}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    
    # Save and log to MLflow
    plot_path = f"confusion_matrix_fold{fold}_epoch{epoch}.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()