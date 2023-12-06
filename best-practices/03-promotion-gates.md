# Practice 3: Promote with Gates, Not Vibes

## The Problem

```
❌ "I'm pretty sure this model is better, let's deploy it"
❌ "The metrics look good to me"  
❌ "YOLO ship it"
```

Humans are optimistic. We remember successes and forget edge cases. We get excited about improvements and overlook regressions. **Gates don't forget.**

## The Solution: Automated Promotion Gates

See the [Full Detailed Guide](FULL_GUIDE.md#practice-3-promote-with-gates-not-vibes) for complete implementation including:

- Full `PromotionGates` class with offline, staging, and production gate checks
- Metric thresholds and fairness validation
- Latency and performance gates  
- CI/CD integration examples
- GitHub Actions workflow

### Quick Implementation Pattern

```python
from mlflow.tracking import MlflowClient

class PromotionGates:
    def __init__(self, client: MlflowClient):
        self.client = client
    
    def check_offline_gates(self, candidate_version, champion_version, model_name):
        """Gates before staging"""
        candidate_metrics = self._get_metrics(model_name, candidate_version)
        champion_metrics = self._get_metrics(model_name, champion_version)
        
        return {
            "accuracy_threshold": candidate_metrics["accuracy"] >= champion_metrics["accuracy"] - 0.02,
            "auc_threshold": candidate_metrics["auc"] >= champion_metrics["auc"] - 0.01,
            "fairness_check": candidate_metrics.get("demographic_parity", 0) <= 0.1
        }
    
    def check_staging_gates(self, model_name, version, staging_metrics):
        """Gates before production"""
        return {
            "latency_p95": staging_metrics.get("p95_latency_ms", 1000) <= 25,
            "latency_p99": staging_metrics.get("p99_latency_ms", 1000) <= 50,
            "memory_usage": staging_metrics.get("memory_mb", 2000) <= 1024,
            "error_rate": staging_metrics.get("error_rate", 1.0) <= 0.001
        }
```

### Gate Categories

**Offline Gates** (Before Staging):
- Metric thresholds vs champion
- Fairness and bias checks
- Feature coverage validation
- Model size constraints

**Staging Gates** (Before Production):
- Latency (P95, P99)
- Memory footprint
- Cold start time
- Error rate
- Logging completeness

**Production Gates** (Final Check):
- A/B test statistical significance
- Business metric lift
- No SLA violations
- Successful shadow period (3+ days)

---

**Why it ages well**: New teammates can ship with confidence because the gates don't forget edge cases that caused past incidents.

---

**Navigation**: [← Previous: Model Signatures](02-model-signatures.md) | [Back to Index](README.md) | [Next: Rollbacks →](04-rollbacks.md) | [Full Guide](FULL_GUIDE.md)
