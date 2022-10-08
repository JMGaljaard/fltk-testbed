variable "kubernetes_config_path" {
  description = "Path of Kubernetes configuration file (change for non-default kubectl setup)"
  default     = "~/.kube/config"
}

variable "project_id" {
  type        = string
  default     = "test-bed-fltk"
  description = "Google Cloud project name to create cluster in."
}

variable "cluster_name" {
  type        = string
  default     = "fltk-testbed-cluster"
  description = "Name of the GKE cluster to be deployed in project <project_id>."
}

variable "project_region" {
  type        = string
  default     = "us-central1"
  description = "GKE region to deploy cluster in."
}

variable "project_zone" {
  type        = string
  default     = "us-central1-c"
  description = "GKE region to deploy cluster in."
}

variable "description" {
  type    = string
  default = "Managed by terraform FLTK testbed deployment"
}

variable "account_id" {
  type        = string
  description = "The service account Identifier to be used to interact with Google cloud."
  default     = "terraform-iam-service-account"
}

variable "kubeflow_version" {
  type        = string
  description = "Kubeflow (training operator) to install."
  default     = "v1.5.0"
}

variable "nfs_provider_information" {
  type = object({
    release_name     = string
    chart_name       = string
    namespace        = string
    server_path      = string
    image_repository = string
    image_tag        = string
    pull_policy      = string
    storage_size     = string
  })
  default = {
    release_name     = "nfs-server"
    chart_name       = "nfs-server-provisioner"
    namespace        = "test"
    server_path      = "/mnt/kubernetes"
    image_repository = "k8s.gcr.io/sig-storage/nfs-provisioner"
    image_tag        = "v3.0.0"
    pull_policy      = "IfNotPresent"
    storage_size     = "50Gi"
  }
}

variable "nfs_provisioner_repo_url" {
  description = "Repository URL to locate the utilized helm charts"
  type        = string
  default     = "https://charts.helm.sh/stable"
}
