variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for resource deployment"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "distilbert-sentiment"
}

variable "gcs_bucket_name" {
  description = "Name of the GCS bucket for datasets and artifacts"
  type        = string
  default     = "distilbert-sentiment-data"
}

variable "artifact_registry_name" {
  description = "Name of the Artifact Registry repository"
  type        = string
  default     = "distilbert-sentiment"
}
