variable "kubernetes_version" {
  type        = string
  description = "Kubernetes version to use, note that kubeflow depends on version pre-1.22 for deployment as of now."
  default     = "1.21"
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

variable "regional_deployment" {
  type        = bool
  default     = false
  description = "Whether to deploy regional (additional 0.10 USD/hour cost for management) for higher availability."
}

variable "project_region" {
  type        = string
  default     = "us-central1"
  description = "GKE region to deploy cluster in."
}

variable "project_zones" {
  type        = list(string)
  default     = ["us-central1-c"]
  description = "GKE zones to deploy in. Zonal deployment will only take first zone."
}

variable "description" {
  type        = string
  description = "Service account description"
  default     = "Managed by terraform FLTK testbed deployment"
}

variable "account_id" {
  description = "The service account Identifier to be used to interact with Google cloud."
  default     = "terraform-iam-service-account"
}

variable "network" {
  description = "Private network to be used by the cluster during deployment."
  default     = "gcp-private-network"
}

variable "subnetwork" {
  description = "Subnetwork name for private network used in the cluster."
  default     = "gcp-private-subnetwork"
}

variable "ip_range_pods_name" {
  description = "The secondary ip range to use for pods"
  default     = "ip-range-pods"
}

variable "ip_range_services_name" {
  description = "The secondary ip range to use for services"
  default     = "ip-range-scv"
}

variable "ip_range_pods_range" {
  description = "The secondary ip range to use for pods"
  default     = "192.168.0.0/18"
}

variable "ip_range_services_range" {
  description = "The secondary ip range to use for services, must not be overlapping with `ip_range_pods_name`"
  default     = "192.168.64.0/18"
}
