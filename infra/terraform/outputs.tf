output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "gcs_bucket_url" {
  description = "GCS bucket URL"
  value       = "gs://${google_storage_bucket.data_bucket.name}"
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository URL"
  value       = "${google_artifact_registry_repository.repo.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}"
}
