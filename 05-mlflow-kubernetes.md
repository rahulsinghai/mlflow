# 05 - MLflow on Kubernetes

## Overview
Running MLflow on Kubernetes provides scalability, high availability, and enterprise-grade deployment capabilities. This guide covers production-ready setups.

## MLflow on Kubernetes Architecture

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        A[Ingress Controller] --> B[MLflow Tracking Server]
        B --> C[(PostgreSQL)]
        B --> D[(S3/MinIO)]
        
        E[Training Jobs] --> B
        F[Model Registry UI] --> B
        
        G[Model Serving Pods] --> D
        
        H[Prometheus] --> B
        H --> G
    end
    
    I[Data Scientists] --> A
    J[CI/CD Pipeline] --> E
    K[Applications] --> G
    
    style B fill:#4A90E2
    style C fill:#50C878
    style D fill:#FFD700
```

## Prerequisites Setup

### 1. PostgreSQL Database
```yaml
# postgres-deployment.yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
type: Opaque
stringData:
  POSTGRES_USER: mlflow
  POSTGRES_PASSWORD: mlflow_password
  POSTGRES_DB: mlflowdb
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        envFrom:
        - secretRef:
            name: postgres-secret
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

### 2. MinIO (S3-compatible storage)
```yaml
# minio-deployment.yaml
apiVersion: v1
kind: Secret
metadata:
  name: minio-secret
type: Opaque
stringData:
  MINIO_ROOT_USER: minio
  MINIO_ROOT_PASSWORD: minio123
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        ports:
        - containerPort: 9000
        - containerPort: 9001
        envFrom:
        - secretRef:
            name: minio-secret
        volumeMounts:
        - name: minio-storage
          mountPath: /data
      volumes:
      - name: minio-storage
        persistentVolumeClaim:
          claimName: minio-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: minio-service
spec:
  selector:
    app: minio
  ports:
  - name: api
    port: 9000
    targetPort: 9000
  - name: console
    port: 9001
    targetPort: 9001
  type: LoadBalancer
```

## MLflow Tracking Server Deployment

### MLflow Server ConfigMap
```yaml
# mlflow-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-config
data:
  create-bucket.sh: |
    #!/bin/sh
    # Create MLflow bucket in MinIO
    mc alias set myminio http://minio-service:9000 minio minio123
    mc mb myminio/mlflow-artifacts --ignore-existing
    echo "Bucket created successfully"
```

### MLflow Deployment
```yaml
# mlflow-deployment.yaml
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-secret
type: Opaque
stringData:
  DB_URI: "postgresql://mlflow:mlflow_password@postgres-service:5432/mlflowdb"
  AWS_ACCESS_KEY_ID: "minio"
  AWS_SECRET_ACCESS_KEY: "minio123"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  labels:
    app: mlflow-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      initContainers:
      - name: wait-for-db
        image: postgres:15
        command:
        - sh
        - -c
        - |
          until pg_isready -h postgres-service -p 5432 -U mlflow; do
            echo "Waiting for database..."
            sleep 2
          done
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.10.0
        ports:
        - containerPort: 5000
          name: http
        env:
        - name: BACKEND_STORE_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-secret
              key: DB_URI
        - name: ARTIFACT_ROOT
          value: "s3://mlflow-artifacts/"
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio-service:9000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: mlflow-secret
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: mlflow-secret
              key: AWS_SECRET_ACCESS_KEY
        command:
        - mlflow
        - server
        - --host=0.0.0.0
        - --port=5000
        - --backend-store-uri=$(BACKEND_STORE_URI)
        - --default-artifact-root=$(ARTIFACT_ROOT)
        - --workers=4
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
spec:
  selector:
    app: mlflow-server
  ports:
  - protocol: TCP
    port: 5000
    targetPort: 5000
  type: LoadBalancer
```

### Ingress Configuration
```yaml
# mlflow-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlflow-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - mlflow.yourdomain.com
    secretName: mlflow-tls
  rules:
  - host: mlflow.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mlflow-service
            port:
              number: 5000
```

## Deploy Everything

```bash
# 1. Create namespace
kubectl create namespace mlflow

# Set context
kubectl config set-context --current --namespace=mlflow

# 2. Deploy PostgreSQL
kubectl apply -f postgres-deployment.yaml

# 3. Deploy MinIO
kubectl apply -f minio-deployment.yaml

# Wait for MinIO to be ready
kubectl wait --for=condition=ready pod -l app=minio --timeout=300s

# 4. Create bucket (one-time setup)
kubectl run -it --rm create-bucket \
  --image=minio/mc \
  --restart=Never \
  -- sh -c "mc alias set myminio http://minio-service:9000 minio minio123 && mc mb myminio/mlflow-artifacts --ignore-existing"

# 5. Deploy MLflow
kubectl apply -f mlflow-deployment.yaml

# 6. Deploy Ingress (optional)
kubectl apply -f mlflow-ingress.yaml

# Check status
kubectl get pods
kubectl get svc
kubectl logs -l app=mlflow-server
```

## Helm Chart Deployment

### Using Community Helm Chart
```bash
# Add Helm repo
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update

# Create values file
cat > mlflow-values.yaml <<EOF
backendStore:
  postgres:
    enabled: true
    host: postgres-service
    port: 5432
    database: mlflowdb
    user: mlflow
    password: mlflow_password

artifactRoot:
  s3:
    enabled: true
    bucket: mlflow-artifacts
    endpointUrl: http://minio-service:9000
    accessKeyId: minio
    secretAccessKey: minio123

service:
  type: LoadBalancer
  port: 5000

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

replicaCount: 2

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: mlflow.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mlflow-tls
      hosts:
        - mlflow.yourdomain.com
EOF

# Install MLflow
helm install mlflow community-charts/mlflow \
  --namespace mlflow \
  --create-namespace \
  -f mlflow-values.yaml

# Upgrade
helm upgrade mlflow community-charts/mlflow \
  -f mlflow-values.yaml
```

## Training Jobs on Kubernetes

### Kubernetes Job for Model Training
```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: iris-training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: your-registry/ml-trainer:latest
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: mlflow-secret
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: mlflow-secret
              key: AWS_SECRET_ACCESS_KEY
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio-service:9000"
        command:
        - python
        - /app/train.py
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      restartPolicy: Never
  backoffLimit: 3
```

### Training Script
```python
# train.py
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("iris-k8s-training")

# Enable autologging
mlflow.sklearn.autolog()

def train():
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Train with MLflow tracking
    with mlflow.start_run(run_name="k8s-training") as run:
        # Log system info
        mlflow.set_tag("training_platform", "kubernetes")
        mlflow.set_tag("job_name", os.getenv("HOSTNAME", "unknown"))
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Run ID: {run.info.run_id}")
        
        # Register model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="iris-classifier-k8s"
        )

if __name__ == "__main__":
    train()
```

### Run Training Job
```bash
# Build and push trainer image
docker build -t your-registry/ml-trainer:latest -f Dockerfile.trainer .
docker push your-registry/ml-trainer:latest

# Submit training job
kubectl apply -f training-job.yaml

# Monitor job
kubectl get jobs
kubectl logs job/iris-training-job -f

# Check MLflow for results
kubectl port-forward svc/mlflow-service 5000:5000
# Open http://localhost:5000
```

## CronJob for Scheduled Training

```yaml
# scheduled-training.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-model-training
spec:
  schedule: "0 2 * * *"  # Every day at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: your-registry/ml-trainer:latest
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-service:5000"
            command:
            - python
            - /app/train.py
            resources:
              requests:
                memory: "4Gi"
                cpu: "2000m"
          restartPolicy: OnFailure
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
```

## Monitoring and Observability

### Prometheus Monitoring
```yaml
# prometheus-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mlflow-metrics
  labels:
    app: mlflow-server
spec:
  selector:
    matchLabels:
      app: mlflow-server
  endpoints:
  - port: http
    interval: 30s
    path: /metrics
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "MLflow Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(mlflow_http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(mlflow_http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Active Runs",
        "targets": [
          {
            "expr": "mlflow_active_runs"
          }
        ]
      }
    ]
  }
}
```

## High Availability Setup

### StatefulSet for MLflow (Alternative)
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow-server
spec:
  serviceName: mlflow-headless
  replicas: 3
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.10.0
        # ... same configuration as Deployment
  volumeClaimTemplates:
  - metadata:
      name: mlflow-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

## Security Best Practices

### Network Policies
```yaml
# mlflow-network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mlflow-network-policy
spec:
  podSelector:
    matchLabels:
      app: mlflow-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: mlflow-client
    ports:
    - protocol: TCP
      port: 5000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: minio
    ports:
    - protocol: TCP
      port: 9000
```

### RBAC Configuration
```yaml
# mlflow-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlflow-sa
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlflow-role
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlflow-rolebinding
subjects:
- kind: ServiceAccount
  name: mlflow-sa
roleRef:
  kind: Role
  name: mlflow-role
  apiGroup: rbac.authorization.k8s.io
```

---
**Navigation**: [← Previous](04-mlflow-deployment.md) | Next → [06-mlflow-zenml-integration.md](06-mlflow-zenml-integration.md)
