# Kustomization (i.e. Kustomize) requires kubernetes config
provider "kustomization" {
  kubeconfig_path = var.kubernetes_config_path
}

provider "kubectl" {
  config_path = var.kubernetes_config_path
}

provider "kubernetes" {
  config_path = kubernetes_config_path
}

provider "helm" {
  kubernetes {
    config_path = var.kubernetes_config_path
  }
}
