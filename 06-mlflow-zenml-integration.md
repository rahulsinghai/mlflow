# 06 - MLflow & ZenML Integration

## Overview
ZenML is an MLOps framework for building portable, production-ready pipelines. Integrating MLflow with ZenML combines pipeline orchestration with experiment tracking and model management.

## Architecture Overview

```mermaid
graph TB
    subgraph "ZenML Pipeline"
        A[Data Ingestion] --> B[Data Validation]
        B --> C[Feature Engineering]
        C --> D[Model Training]
        D --> E[Model Evaluation]
        E --> F[Model Deployment]
    end
    
    subgraph "MLflow Integration"
        D --> G[MLflow Tracking]
        D --> H[MLflow Models]
        E --> G
        F --> I[MLflow Registry]
    end
    
    subgraph "Infrastructure"
        G --> J[(Tracking Server)]
        H --> K[(Artifact Store)]
        I --> L[Model Registry]
    end
    
    M[ZenML Server] -.manages.-> A
    M -.manages.-> B
    M -.manages.-> C
    
    style D fill:#4A90E2
    style G fill:#FFD700
    style I fill:#50C878
```

## Installation & Setup

### Install Dependencies
```bash
# Install ZenML with MLflow integration
pip install "zenml[server]" mlflow

# Initialize ZenML
zenml init

# Install MLflow integration
zenml integration install mlflow -y

# Start ZenML server (optional but recommended)
zenml up

# Access ZenML dashboard at http://localhost:8237
```

### Configure MLflow with ZenML

```python
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

# Get ZenML client
client = Client()

# Register MLflow experiment tracker
client.create_stack_component(
    name="mlflow_tracker",
    flavor="mlflow",
    component_type="experiment_tracker",
    configuration={
        "tracking_uri": "http://mlflow-server:5000",
        "tracking_username": "",  # Optional
        "tracking_password": "",  # Optional
    }
)

# Register MLflow model deployer
client.create_stack_component(
    name="mlflow_deployer",
    flavor="mlflow",
    component_type="model_deployer",
    configuration={
        "tracking_uri": "http://mlflow-server:5000",
    }
)

# Create and set active stack
client.create_stack(
    name="mlflow_stack",
    components={
        "experiment_tracker": "mlflow_tracker",
        "model_deployer": "mlflow_deployer",
        "orchestrator": "default",
        "artifact_store": "default"
    }
)

client.activate_stack("mlflow_stack")
```

## Basic ZenML Pipeline with MLflow

### Simple Training Pipeline
```python
from zenml import pipeline, step
from zenml.client import Client
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Data loading step
@step
def load_data() -> tuple:
    """Load and split iris dataset"""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Training step with MLflow tracking
@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5
) -> RandomForestClassifier:
    """Train model with MLflow tracking"""
    
    # MLflow is automatically configured by ZenML
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Log custom parameters
    mlflow.log_param("zenml_pipeline", "iris_training")
    
    return model

# Evaluation step
@step(experiment_tracker="mlflow_tracker")
def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """Evaluate model and log metrics to MLflow"""
    
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    # Log to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    return {
        "accuracy": accuracy,
        "f1_score": f1
    }

# Define pipeline
@pipeline(enable_cache=False)
def training_pipeline(
    n_estimators: int = 100,
    max_depth: int = 5
):
    """Complete training pipeline with MLflow integration"""
    
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, n_estimators, max_depth)
    metrics = evaluate_model(model, X_test, y_test)
    
    return model, metrics

# Run pipeline
if __name__ == "__main__":
    training_pipeline(n_estimators=100, max_depth=10)
```

## Advanced Pipeline with Model Registry

```python
from zenml import pipeline, step
from zenml.client import Client
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.logger import get_logger

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Tuple, Dict
import pandas as pd
import numpy as np

logger = get_logger(__name__)

@step
def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess data"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    iris = load_iris()
    
    # Create DataFrame
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test

@step(experiment_tracker="mlflow_tracker")
def train_and_register_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "iris-zenml-classifier",
    n_estimators: int = 100,
    max_depth: int = 5
) -> str:
    """Train model and register in MLflow"""
    
    from sklearn.ensemble import RandomForestClassifier
    from mlflow.models import infer_signature
    
    # Enable autologging
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
    
    with mlflow.start_run() as run:
        # Log pipeline info
        mlflow.set_tag("pipeline", "zenml_training")
        mlflow.set_tag("framework", "zenml+mlflow")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Infer signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Register model
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name=model_name,
            input_example=X_train.iloc[:5]
        )
        
        run_id = run.info.run_id
        logger.info(f"Model registered with run_id: {run_id}")
        
        return run_id

@step(experiment_tracker="mlflow_tracker")
def evaluate_and_promote(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_id: str,
    model_name: str = "iris-zenml-classifier",
    accuracy_threshold: float = 0.90
) -> Dict[str, float]:
    """Evaluate model and promote to staging/production"""
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Load model from run
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted'),
        "recall": recall_score(y_test, predictions, average='weighted'),
        "f1_score": f1_score(y_test, predictions, average='weighted')
    }
    
    # Log evaluation metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"eval_{metric_name}", metric_value)
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Get model version
    model_versions = client.search_model_versions(f"run_id='{run_id}'")
    if model_versions:
        version = model_versions[0].version
        
        # Promote based on accuracy
        if metrics["accuracy"] >= accuracy_threshold:
            # Promote to staging first
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            logger.info(f"✓ Model version {version} promoted to Staging")
            
            # Tag for approval
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key="approval_status",
                value="pending_production"
            )
        else:
            logger.warning(f"✗ Model accuracy {metrics['accuracy']:.4f} below threshold {accuracy_threshold}")
    
    return metrics

@step(experiment_tracker="mlflow_tracker")
def deploy_model(
    model_name: str = "iris-zenml-classifier",
    stage: str = "Staging"
) -> str:
    """Deploy model from MLflow registry"""
    
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Get latest version in stage
    versions = client.get_latest_versions(model_name, stages=[stage])
    
    if not versions:
        raise ValueError(f"No model found in stage: {stage}")
    
    version = versions[0].version
    model_uri = f"models:/{model_name}/{stage}"
    
    logger.info(f"Deploying model {model_name} version {version} from {stage}")
    
    # Here you would typically deploy to your serving infrastructure
    # For now, we'll just log the deployment
    mlflow.set_tag("deployed_model", model_name)
    mlflow.set_tag("deployed_version", version)
    mlflow.set_tag("deployed_stage", stage)
    
    return model_uri

# Complete pipeline
@pipeline(enable_cache=False)
def ml_pipeline(
    model_name: str = "iris-zenml-classifier",
    n_estimators: int = 100,
    max_depth: int = 5,
    accuracy_threshold: float = 0.90
):
    """End-to-end ML pipeline with ZenML and MLflow"""
    
    # Load and preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train and register
    run_id = train_and_register_model(
        X_train, y_train, model_name, n_estimators, max_depth
    )
    
    # Evaluate and promote
    metrics = evaluate_and_promote(
        X_test, y_test, run_id, model_name, accuracy_threshold
    )
    
    # Deploy if approved
    model_uri = deploy_model(model_name, stage="Staging")
    
    return metrics, model_uri

# Run pipeline
if __name__ == "__main__":
    pipeline_run = ml_pipeline(
        model_name="iris-zenml-classifier",
        n_estimators=100,
        max_depth=10,
        accuracy_threshold=0.90
    )
```

## ZenML Pipeline on Kubernetes

### Configure Kubernetes Orchestrator
```python
from zenml.client import Client

client = Client()

# Register Kubernetes orchestrator
client.create_stack_component(
    name="k8s_orchestrator",
    flavor="kubernetes",
    component_type="orchestrator",
    configuration={
        "kubernetes_context": "your-k8s-context",
        "kubernetes_namespace": "zenml",
        "synchronous": False
    }
)

# Update stack
client.update_stack(
    name="mlflow_stack",
    components={
        "orchestrator": "k8s_orchestrator",
        "experiment_tracker": "mlflow_tracker",
        "model_deployer": "mlflow_deployer",
        "artifact_store": "default"
    }
)
```

### Pipeline with Resource Requirements
```python
from zenml import pipeline, step
from zenml.config import ResourceSettings

# Configure resources for training step
@step(
    experiment_tracker="mlflow_tracker",
    settings={
        "orchestrator.kubernetes": ResourceSettings(
            cpu_count=4,
            memory="8GB",
            gpu_count=1  # If using GPU
        )
    }
)
def train_large_model(X_train, y_train):
    """Training step with specific resource requirements"""
    
    mlflow.sklearn.autolog()
    
    # Train computationally intensive model
    model = train_heavy_model(X_train, y_train)
    
    return model

@pipeline
def k8s_training_pipeline():
    """Pipeline that runs on Kubernetes"""
    
    X_train, X_test, y_train, y_test = load_data()
    model = train_large_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    return model, metrics
```

## Hyperparameter Tuning with ZenML & MLflow

```python
from zenml import pipeline, step
import mlflow
from typing import List, Dict
import optuna

@step(experiment_tracker="mlflow_tracker")
def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50
) -> Dict:
    """Hyperparameter tuning with Optuna and MLflow"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        
        # Create nested run for each trial
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            val_predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_predictions)
            
            # Log metric
            mlflow.log_metric("val_accuracy", accuracy)
            
            return accuracy
    
    # Run optimization
    with mlflow.start_run(run_name="hyperparameter_tuning"):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Log best parameters
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_accuracy", study.best_value)
        
        # Log optimization history
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        mlflow.log_figure(fig, "optimization_history.png")
        
        return study.best_params

@pipeline(enable_cache=False)
def tuning_pipeline(n_trials: int = 50):
    """Pipeline with hyperparameter tuning"""
    
    X_train, X_test, y_train, y_test = load_data()
    
    # Further split training for validation
    from sklearn.model_selection import train_test_split
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Tune hyperparameters
    best_params = hyperparameter_tuning(
        X_train_sub, y_train_sub, X_val, y_val, n_trials
    )
    
    # Train final model with best parameters
    model = train_model(X_train, y_train, **best_params)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    return model, metrics
```

## ZenML Stack on Kubernetes

```yaml
# zenml-k8s-stack.yaml
# Complete stack configuration for Kubernetes

# MLflow tracking server (already deployed from lesson 05)
# Reference: mlflow-service in namespace mlflow

# ZenML server deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zenml-server
  namespace: zenml
spec:
  replicas: 1
  selector:
    matchLabels:
      app: zenml-server
  template:
    metadata:
      labels:
        app: zenml-server
    spec:
      containers:
      - name: zenml
        image: zenmldocker/zenml-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: ZENML_STORE_URL
          value: "postgresql://zenml:password@postgres-service:5432/zenml"
        - name: ZENML_DEFAULT_PROJECT_NAME
          value: "default"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: zenml-service
  namespace: zenml
spec:
  selector:
    app: zenml-server
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

---
**Navigation**: [← Previous](05-mlflow-kubernetes.md) | Next → [07-mlflow-seldon-integration.md](07-mlflow-seldon-integration.md)
