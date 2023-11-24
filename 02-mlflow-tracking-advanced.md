# 02 - MLflow Tracking Advanced

## Overview
Advanced tracking techniques including nested runs, autologging, custom metrics, and tracking server configuration.

## Tracking Architecture

```mermaid
graph LR
    A[ML Code] --> B[MLflow Client]
    B --> C[Tracking Server]
    C --> D[(Backend Store)]
    C --> E[(Artifact Store)]
    
    D --> D1[SQLite]
    D --> D2[PostgreSQL]
    D --> D3[MySQL]
    
    E --> E1[Local FS]
    E --> E2[S3]
    E --> E3[Azure Blob]
    E --> E4[GCS]
```

## Backend Store Configuration

### SQLite (Default)
```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

### PostgreSQL
```python
mlflow.set_tracking_uri(
    "postgresql://user:password@localhost:5432/mlflowdb"
)
```

### Remote Tracking Server
```python
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

## Artifact Store Configuration

```python
# S3
mlflow.set_tracking_uri("http://mlflow-server:5000")
# Artifacts will use S3 if server configured with --default-artifact-root

# Local with custom path
import mlflow
mlflow.set_tracking_uri("file:///path/to/mlruns")
```

## Autologging

Automatic logging for popular frameworks without explicit log statements.

### Scikit-learn
```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Enable autologging
mlflow.sklearn.autolog()

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)

with mlflow.start_run():
    # Automatically logs parameters, metrics, and model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    # Autolog captures:
    # - All hyperparameters
    # - Training metrics
    # - Model signature
    # - Model artifacts
```

### PyTorch
```python
import mlflow.pytorch
import torch
import torch.nn as nn

mlflow.pytorch.autolog()

with mlflow.start_run():
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Training code automatically logged
    # Logs: model architecture, parameters, loss curves
```

### TensorFlow/Keras
```python
import mlflow.tensorflow
import tensorflow as tf

mlflow.tensorflow.autolog()

with mlflow.start_run():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fit automatically logs everything
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

### XGBoost
```python
import mlflow.xgboost
import xgboost as xgb

mlflow.xgboost.autolog()

with mlflow.start_run():
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'multi:softprob',
        'num_class': 3
    }
    
    # Automatically logs params, metrics, and model
    model = xgb.train(params, dtrain, num_boost_round=100)
```

## Nested Runs

For complex workflows with multiple stages.

```python
import mlflow

# Parent run for entire pipeline
with mlflow.start_run(run_name="full-pipeline") as parent_run:
    mlflow.log_param("pipeline_version", "v1.0")
    
    # Data preprocessing run
    with mlflow.start_run(run_name="preprocessing", nested=True):
        mlflow.log_param("scaling", "standard")
        mlflow.log_param("feature_count", 20)
        # ... preprocessing code ...
    
    # Model training run
    with mlflow.start_run(run_name="training", nested=True):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("train_accuracy", 0.95)
        # ... training code ...
    
    # Model evaluation run
    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_metric("test_accuracy", 0.93)
        mlflow.log_metric("f1_score", 0.92)
        # ... evaluation code ...
    
    # Log overall pipeline metrics
    mlflow.log_metric("pipeline_duration", 120.5)
```

## Custom Metrics and Step-wise Logging

```python
import mlflow
import time

with mlflow.start_run():
    # Log metrics at different steps (epochs, iterations)
    for epoch in range(10):
        train_loss = train_epoch()  # Your training function
        val_loss = validate_epoch()  # Your validation function
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Log learning rate changes
        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)

# This creates beautiful line plots in MLflow UI
```

## Model Signatures

Define input/output schema for models.

```python
from mlflow.models import infer_signature
import pandas as pd
import numpy as np

# Infer signature from data
X_sample = X_train[:5]
predictions = model.predict(X_sample)

signature = infer_signature(X_sample, predictions)

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature
    )

# Manual signature definition
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
    ColSpec("double", "feature_1"),
    ColSpec("double", "feature_2"),
    ColSpec("string", "category")
])

output_schema = Schema([ColSpec("long")])

from mlflow.models import ModelSignature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

## Input Examples

```python
import mlflow
import pandas as pd

input_example = pd.DataFrame({
    "feature_1": [1.0, 2.0, 3.0],
    "feature_2": [4.0, 5.0, 6.0],
    "category": ["A", "B", "C"]
})

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=input_example
    )
```

## Custom Model Logging

```python
import mlflow
from mlflow.pyfunc import PythonModel

class CustomModel(PythonModel):
    def __init__(self, preprocessing_pipeline, model):
        self.preprocessing = preprocessing_pipeline
        self.model = model
    
    def predict(self, context, model_input):
        # Custom preprocessing
        processed = self.preprocessing.transform(model_input)
        # Prediction
        return self.model.predict(processed)

# Log custom model
with mlflow.start_run():
    custom_model = CustomModel(preprocessor, trained_model)
    
    mlflow.pyfunc.log_model(
        artifact_path="custom_model",
        python_model=custom_model,
        pip_requirements=["scikit-learn==1.3.0", "pandas==2.0.0"]
    )
```

## Search and Query Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs by filter
runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Complex filtering
runs = client.search_runs(
    experiment_ids=["1", "2"],
    filter_string="params.model_type = 'RandomForest' AND metrics.accuracy > 0.85",
    order_by=["metrics.f1_score DESC"]
)

# Get run details
for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {run.data.metrics['accuracy']}")
    print(f"Parameters: {run.data.params}")
```

## Comparison and Analysis

```python
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("my-experiment")
runs = client.search_runs(experiment.experiment_id)

# Create comparison dataframe
comparison_data = []
for run in runs:
    comparison_data.append({
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
        **run.data.params,
        **run.data.metrics
    })

df = pd.DataFrame(comparison_data)
print(df.sort_values("accuracy", ascending=False))
```

## Logging Artifacts at Scale

```python
import mlflow
import os

with mlflow.start_run():
    # Log single file
    mlflow.log_artifact("model.pkl")
    
    # Log entire directory
    mlflow.log_artifacts("output_dir", artifact_path="outputs")
    
    # Log dictionary as JSON
    mlflow.log_dict({"key": "value"}, "config.json")
    
    # Log text
    mlflow.log_text("Model training completed", "status.txt")
    
    # Log figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    mlflow.log_figure(fig, "plot.png")
```

## System Metrics

```python
import mlflow

# Log system metrics during training
mlflow.enable_system_metrics_logging()

with mlflow.start_run():
    # Automatically logs:
    # - CPU usage
    # - Memory usage
    # - GPU usage (if available)
    # - Disk I/O
    # - Network I/O
    
    train_model()  # Your training code
```

## Run Context and Metadata

```python
import mlflow
from mlflow.entities import RunStatus

with mlflow.start_run() as run:
    # Access run info
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Start time: {run.info.start_time}")
    
    # Set tags
    mlflow.set_tags({
        "team": "ml-team",
        "priority": "high",
        "model_version": "v2.0",
        "git_commit": "abc123"
    })
    
    # Log environment info
    import platform
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("system", platform.system())
```

---
**Navigation**: [← Previous](01-mlflow-fundamentals.md) | Next → [03-mlflow-models-registry.md](03-mlflow-models-registry.md)
