# Browser Version and YC Deployment

## Local run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start app:

```bash
streamlit run app/web/streamlit_app.py
```

3. Open URL from Streamlit output (usually `http://localhost:8501`).

## Docker run

```bash
docker build -t scan-visualizer:local .
docker run --rm -p 8501:8501 scan-visualizer:local
```

## Kubernetes in Yandex Cloud

Use manifests from `k8s/`:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

Before deploy, set your image in `k8s/deployment.yaml`:

- `cr.yandex/<registry-id>/scan-visualizer:latest`

and host in `k8s/ingress.yaml`:

- `scan-visualizer.example.com`
