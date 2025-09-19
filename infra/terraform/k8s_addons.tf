# Configure Helm provider to use our GKE cluster
provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
  }
}

# Install Ingress-NGINX controller
resource "helm_release" "ingress_nginx" {
  name             = "ingress-nginx"
  repository       = "https://kubernetes.github.io/ingress-nginx"
  chart            = "ingress-nginx"
  namespace        = "ingress-nginx"
  create_namespace = true
  version          = "4.8.3"  # Specify a version for stability

  values = [
    jsonencode({
      controller = {
        service = {
          type = "LoadBalancer"
        }
        # Use a custom namespace for resources
        resources = {
          namespace = "ingress-nginx"
        }
      }
    })
  ]
}

# Install cert-manager
resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  namespace        = "cert-manager"
  create_namespace = true
  version          = "v1.13.3"  # Specify a version for stability

  values = [
    jsonencode({
      installCRDs = true,
      # Avoid using kube-system namespace
      global = {
        leaderElection = {
          namespace = "cert-manager"
        }
      }
    })
  ]
}

# Install kube-prometheus-stack
resource "helm_release" "kube_prometheus_stack" {
  name             = "kube-prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
  version          = "55.5.0"  # Specify a version for stability

  values = [
    jsonencode({
      alertmanager = {
        enabled = true
      }
      grafana = {
        enabled = true
      }
      prometheus = {
        enabled = true
      }
      # Avoid using kube-system namespace
      prometheusOperator = {
        namespaces = {
          additional = ["monitoring"]
        }
      }
      # Use custom namespace for resources
      commonNamespace = "monitoring"
    })
  ]
}
