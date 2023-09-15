variable "kubernetes_config_path" {
  description = "Path of Kubernetes configuration file (change for non-default kubectl setup)"
  default     = "~/.kube/config"
}

variable "kubeflow_version" {
  type        = string
  description = "Kubeflow (PyTorch) Training Operator to install."
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

variable "vulcano_scheduler_information" {
  type = object({
    release_name     = string
    chart_name       = string
    namespace        = string
    version          = string
  })
  default = {
    release_name     = "volcano"
    chart_name       = "volcano"
    namespace        = "volcano-system"
    version          = "v1.8.0"
  }
}

variable "vulcano_scheduler_repo_url" {
  description = "Repository URL to locate the utilized helm charts for Vulcano Scheduler Plugin."
  type        = string
  default     = "https://volcano-sh.github.io/helm-charts"
}

