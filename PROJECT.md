# SentimentOps: DistilBERT Sentiment Analysis with End-to-End MLOps

## Overview
SentimentOps is an end-to-end MLOps project that demonstrates how to train, deploy, monitor, and continuously improve a **DistilBERT** sentiment analysis model.  
It combines popular open-source tools with cloud-native infrastructure to showcase best practices for building reliable and scalable machine learning systems.

## Key Features
- **Data Management & Versioning:** Use [DVC](https://dvc.org/) to version the IMDB dataset and store large artifacts in GCS.
- **Experiment Tracking & Registry:** Log runs, metrics, and artifacts with [MLflow](https://mlflow.org/); manage model versions via its Model Registry.
- **Training on GPUs:** Fine-tune DistilBERT on GKE GPU nodes for efficient NLP training.
- **Pipelines & Automation:** Orchestrate training, evaluation, and deployment workflows using [Prefect](https://www.prefect.io/) or [Argo Workflows](https://argoproj.github.io/).
- **Model Serving:** Package and serve the model with [BentoML](https://bentoml.com/) (or [KServe](https://kserve.github.io/)) on Kubernetes.
- **Monitoring & Drift Detection:** Track latency, error rates, and data drift using [Prometheus](https://prometheus.io/), [Grafana](https://grafana.com/), and [Evidently](https://evidentlyai.com/).
- **CI/CD:** Automate builds, tests, and deployments with GitHub Actions; promote new models through canary rollouts.
- **Security & Cost Control:** Use GCP Workload Identity, RBAC, and autoscaled GPU node pools with TTL-cleanup for Jobs.

## Architecture
1. **Data Ingestion & Validation**  
   - IMDB dataset loaded from Hugging Face.  
   - Schema & quality checks with Great Expectations.  
   - Version snapshots stored via DVC (remote: GCS).

2. **Experimentation & Training**  
   - Fine-tune DistilBERT using PyTorch + Transformers.  
   - Train on GPUs with mixed precision (FP16).  
   - Log metrics and checkpoints to MLflow.

3. **Evaluation & Model Registry**  
   - Evaluate accuracy and F1-score.  
   - Register models in MLflow if they exceed baseline.

4. **Packaging & Deployment**  
   - Wrap the model in BentoML; build a container image.  
   - Deploy to GKE with autoscaling and health probes.

5. **Monitoring & Alerts**  
   - Expose metrics for Prometheus.  
   - Grafana dashboards for latency, throughput, drift PSI.  
   - Alertmanager sends Slack alerts for anomalies.

6. **Retraining & Continuous Delivery**  
   - Prefect or Argo retraining pipeline triggered by drift or schedule.  
   - Canary rollout for new models; auto-promotion after soak.

---

## Stages

### Stage 1: Project Initialization
- Create a GitHub repository and scaffold the folder structure (`app/`, `data/`, `infra/`, `flows/`, `monitoring/`).
- Write down SLOs (latency, accuracy, availability, cost) in the README.
- Set up basic GitHub Actions (lint/test placeholder).
- Provision a GCP project, enable APIs (GKE, GCS, Artifact Registry).

### Stage 2: Infrastructure Setup
- Use Terraform to create:
  - GKE cluster (Autopilot or Standard + GPU node pool).
  - GCS bucket for datasets & artifacts.
  - Artifact Registry for container images.
- Configure Workload Identity for service-to-service auth.
- Install cluster add-ons:
  - Ingress-NGINX and cert-manager.
  - kube-prometheus-stack (Prometheus + Grafana).
  - NVIDIA device plugin (if using Standard GKE GPUs).

### Stage 3: Data Management
- Fetch a sample of the IMDB dataset from Hugging Face.
- Version data with **DVC**, store large files in GCS.
- Implement validation rules with **Great Expectations** (schema, missing values, label balance).

### Stage 4: Model Development & Training
- Write the training script for DistilBERT (PyTorch + Transformers).
- Integrate MLflow for experiment logging (metrics, hyperparameters, artifacts).
- Run training locally or in a GPU-enabled K8s Job.
- Define a baseline accuracy/F1.

### Stage 5: Model Evaluation & Registration
- Evaluate the trained model against the baseline.
- Register models in MLflow Model Registry.
- Define promotion policies (Staging → Production).

### Stage 6: Packaging & Serving
- Create a BentoML service wrapping the model with `/predict`, `/health`, `/metrics`.
- Build and push the Bento image to Artifact Registry.
- Deploy to GKE:
  - Deployment + Service + HPA.
  - Ingress with TLS.
  - Secure access via Workload Identity and RBAC.

### Stage 7: Monitoring & Drift Detection
- Expose application metrics (latency, errors, version) for Prometheus.
- Build Grafana dashboards.
- Schedule an Evidently CronJob to compute drift metrics (PSI, KS).
- Configure Alertmanager → Slack for SLO violations or drift.

### Stage 8: CI/CD & Automation
- Expand GitHub Actions:
  - Run unit tests, style checks, and smoke inference.
  - Build/push container on merge to `main`.
  - Deploy canary to GKE; promote after soak if metrics stable.
- Add a Prefect (or Argo) pipeline:
  - Ingest new data → validate → retrain on GPU → evaluate → register → deploy.

### Stage 9: Documentation & Runbook
- Write a runbook:
  - How to retrain and deploy.
  - How to read dashboards and handle alerts.
  - Rollback procedures for bad canaries.
- Add diagrams:
  - High-level architecture.
  - Data flow and retraining pipeline.

---

## Deliverables
- A GitHub repository with:
  - Training, inference, and pipeline code.
  - Terraform and Kubernetes manifests.
  - CI/CD workflows and monitoring configs.
- Deployed sentiment analysis API with `/predict`, `/health`, and `/metrics`.
- Grafana dashboards and example Slack alerts.
- Documented runbook with SLOs, rollback steps, and architecture diagrams.

## Goals
- Showcase an industry-grade MLOps workflow.
- Demonstrate GPU-enabled NLP training.
- Provide a template for production-ready ML services.
