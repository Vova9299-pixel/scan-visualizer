# Deploy to Yandex Cloud Kubernetes

1. Build and publish image to GHCR (automatic):

```bash
git push origin master
```

The workflow publishes:

- `ghcr.io/vova9299-pixel/scan-visualizer:latest`

2. Ensure image path in `k8s/deployment.yaml` is correct.

3. Deploy manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

4. Check rollout:

```bash
kubectl rollout status deploy/scan-visualizer
kubectl get pods -l app=scan-visualizer
kubectl get ingress scan-visualizer
```
