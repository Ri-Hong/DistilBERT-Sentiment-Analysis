resource "google_storage_bucket" "data_bucket" {
  name          = var.gcs_bucket_name
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30  # days
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90  # days
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
}

# IAM binding for the GKE service account to access the bucket
resource "google_storage_bucket_iam_member" "gke_sa_storage_access" {
  bucket = google_storage_bucket.data_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gke_sa.email}"
}
