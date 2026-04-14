# Deploy to Yandex Cloud Kubernetes

1. Build image:

```bash
docker build -t cr.yandex/<registry-id>/scan-visualizer:latest .
```

2. Push image:

```bash
docker push cr.yandex/<registry-id>/scan-visualizer:latest
```

3. Update image path in `k8s/deployment.yaml`.

4. Deploy manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

5. Check rollout:

```bash
kubectl rollout status deploy/scan-visualizer
kubectl get pods -l app=scan-visualizer
kubectl get ingress scan-visualizer
```
