variable "kubernetes_config_path" {
  description = "Path of Kubernetes configuration file"
  default     = "~/.kube/config"
}

variable "project_id" {
  type = string
  default = "qpecs-fltk-2022"
  description = "Google Cloud project name to create cluster in."
}

variable "cluster_name" {
  type = string
  default = "freddie-testbed-cluster"
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
  description = "The service account Identifier to be used to interact with Google cloud."
  default = "terraform-iam-service-account"
}

