# Practice 5: Log the Full Story

## What to Log

Models are more than binaries. Capture complete context:

### 1. Lineage
- Dataset commit SHA
- Feature view versions
- Code package version
- Parent model IDs (for ensembles)

### 2. Parameters
- Hyperparameters
- Feature toggles
- Preprocessing config

### 3. Artifacts
- Feature importance plots
- Calibration curves
- Confusion matrices
- SHAP summaries
- Model cards

### 4. Decision Context
- Why this version was promoted
- Business context
- Approver
- Risk assessment

## Quick Implementation

```python
import mlflow

lineage = {
    "git_commit": "ab12cd3",
    "dataset_commit": "data-v2.1",
    "feature_views": {"txn": "v5", "geo": "v3"}
}

decision_context = {
    "reason": "Improved recall on fraud type X",
    "approver": "@meera",
    "risks": "Slight FP increase on low-value txns"
}

with mlflow.start_run():
    # Log lineage
    mlflow.log_dict(lineage, "lineage.json")
    
    # Log decision
    mlflow.log_dict(decision_context, "decision.json")
    
    # Log artifacts
    mlflow.log_figure(importance_plot, "feature_importance.png")
    mlflow.log_figure(confusion_matrix_plot, "confusion_matrix.png")
```

See [Full Guide](FULL_GUIDE.md#practice-5-log-the-full-story) for complete `ComprehensiveLogger` class.

---

**Why it ages well**: Months later, reproduce or legally defend what shipped and why.

---

**Navigation**: [← Previous: Rollbacks](04-rollbacks.md) | [Back to Index](README.md) | [Next: Model/Policy Separation →](06-model-policy-separation.md)
