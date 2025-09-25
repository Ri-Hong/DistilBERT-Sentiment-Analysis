resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region
  project  = var.project_id

  # Enable Autopilot mode
  enable_autopilot    = true
  deletion_protection = false

  # Network configuration
  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {}

  # Enable Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Release channel configuration
  release_channel {
    channel = "REGULAR"
  }
}

# IAM binding for Workload Identity
resource "google_service_account" "gke_sa" {
  account_id   = "${var.cluster_name}-sa"
  display_name = "Service Account for ${var.cluster_name} GKE cluster"
}

resource "google_project_iam_member" "gke_sa_roles" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/artifactregistry.reader"
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.gke_sa.email}"
}
