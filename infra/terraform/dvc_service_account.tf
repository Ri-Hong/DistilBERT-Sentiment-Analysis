# Service account for DVC to access GCS
resource "google_service_account" "dvc_storage" {
  account_id   = "dvc-storage-sa"
  display_name = "DVC Storage Service Account"
  description  = "Service account for DVC to store data in GCS"
}

# Grant storage access to the bucket
resource "google_storage_bucket_iam_member" "dvc_storage_access" {
  bucket = google_storage_bucket.data_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.dvc_storage.email}"
}

# Additional role for writing/updating objects
resource "google_storage_bucket_iam_member" "dvc_storage_writer" {
  bucket = google_storage_bucket.data_bucket.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.dvc_storage.email}"
}

# Create service account key
resource "google_service_account_key" "dvc_storage_key" {
  service_account_id = google_service_account.dvc_storage.name
}

# Output the key (will be used to configure DVC)
output "dvc_service_account_key" {
  value     = google_service_account_key.dvc_storage_key.private_key
  sensitive = true
}
