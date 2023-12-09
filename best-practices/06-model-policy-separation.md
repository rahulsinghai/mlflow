# Practice 6: Separate Model from Policy

## The Anti-Pattern

```python
# ❌ DON'T: Hard-code business logic in model
class FraudModel:
    def predict(self, transaction):
        proba = self.model.predict_proba(transaction)[0, 1]
        if proba > 0.7:  # Threshold hard-coded!
            return "block"
```

**Problems**: Need redeploy to change threshold, can't A/B test, no audit trail.

## The Solution: External Policy

```yaml
# config/fraud_policy.yaml
fraud_detector:
  thresholds:
    block: 0.7
    manual_review: 0.4
  rules:
    high_value_threshold: 10000
```

```python
class PolicyAwareModel:
    def load_context(self, context):
        self.model = mlflow.sklearn.load_model(...)
        self.policy = self._load_policy()  # External config
    
    def predict(self, context, model_input):
        proba = self.model.predict_proba(model_input)
        # Apply policy rules loaded from config
        return self._apply_policy(proba, self.policy)
```

## Benefits

- Change thresholds **without retraining**
- A/B test policies easily
- Audit trail for policy changes
- Governance-friendly
- On-call engineer can tune without ML expertise

See [Full Guide](FULL_GUIDE.md#practice-6-separate-model-from-policy) for complete implementation.

---

**Why it ages well**: Change policy without retraining. Keeps governance happy and on-call sane.

---

**Navigation**: [← Previous: Complete Logging](05-complete-logging.md) | [Back to Index](README.md) | [Next: Shadow & Ramp →](07-shadow-and-ramp.md)
