locals {
  terraform_service_account = "${var.account_id}@${var.project_id}.iam.gserviceaccount.com"
}

provider "google" {
  alias = "impersonation"
  scopes = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
  ]
}

data "google_service_account_access_token" "default" {
  provider               = google.impersonation
  target_service_account = local.terraform_service_account
  scopes                 = ["userinfo-email", "cloud-platform"]
  lifetime               = "1200s"
}

provider "google" {
  project         = var.project_id
  access_token    = data.google_service_account_access_token.default.access_token
  request_timeout = "60s"
}

# Kustomization (i.e. Kustomize) requires kubernetes config
provider "kustomization" {
  kubeconfig_path = var.kubernetes_config_path
}

provider "kubectl" {
  config_path = var.kubernetes_config_path
}


data "google_container_cluster" "testbed_cluster" {
  project  = var.project_id
  name     = var.cluster_name
  location = var.project_zone
}

provider "kubernetes" {
  host  = "https://${data.google_container_cluster.testbed_cluster.endpoint}"
  token = data.google_client_config.default.access_token # Provided by Google data object
  cluster_ca_certificate = base64decode(
    data.google_container_cluster.testbed_cluster.master_auth[0].cluster_ca_certificate,
  )
}


provider "helm" {
  kubernetes {
    config_path = var.kubernetes_config_path
  }

}
