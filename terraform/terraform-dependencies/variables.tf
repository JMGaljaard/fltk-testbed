variable "kubernetes_config_path" {
  description = "Path of Kubernetes configuration file (change for non-default kubectl setup)"
  default     = "~/.kube/config"
}

variable "project_id" {
  type = string
  default = "qpecs-fltk-2022"
  description = "Google Cloud project name to create cluster in."
}

variable "cluster_name" {
  type = string
  default = "fltk-testbed-cluster"
  description = "Name of the GKE cluster to be deployed in project <project_id>."
}

variable "project_region" {
  type = string
  default = "us-central1"
  description = "GKE region to deploy cluster in."
}

variable "description" {
  type = string
  default = "Managed by terraform FLTK testbed deployment"
}

variable "account_id" {
  type = string
  description = "The service account Identifier to be used to interact with Google cloud."
  default = "terraform-iam-service-account"
}

variable "kubeflow_version" {
  type = string
  description = "Kubeflow (training operator) to install."
  default = "v1.5.0"
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
    release_name     = "nfs-client-provisioner"
    chart_name       = "nfs-client-provisioner"
    namespace        = "test"
    server_path      = "/mnt/kubernetes"
    image_repository = "quay.io/external_storage/nfs-client-provisioner"
    image_tag        = "v3.1.0-k8s1.11"
    pull_policy      = "IfNotPresent"
    storage_size     = "50"
  }
}

variable "nfs_provisioner_repo_url" {
  description = "Repository URL to locate the utilized helm charts"
  type        = string
  default     = "https://kvaps.github.io/charts"
}
