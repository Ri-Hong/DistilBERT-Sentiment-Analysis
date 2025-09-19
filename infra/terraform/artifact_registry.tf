resource "google_artifact_registry_repository" "repo" {
  provider = google-beta

  location      = var.region
  repository_id = var.artifact_registry_name
  description   = "Docker repository for DistilBERT Sentiment Analysis containers"
  format        = "DOCKER"

  docker_config {
    immutable_tags = true
  }
}

# IAM binding for the GKE service account to pull images
resource "google_artifact_registry_repository_iam_member" "gke_sa_registry_access" {
  provider = google-beta

  location   = google_artifact_registry_repository.repo.location
  repository = google_artifact_registry_repository.repo.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.gke_sa.email}"
}
