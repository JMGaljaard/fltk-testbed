
data "google_client_config" "default" {}

data "google_container_cluster" "testbed_cluster" {
  name     = var.cluster_name
  location = var.project_region
}

data "kustomization_build" "training-operator" {
  path = "github.com/kubeflow/manifests.git/apps/training-operator/upstream/overlays/standalone?ref=${var.kubeflow_version}"
}



resource "kustomization_resource" "training-operator" {
  for_each = data.kustomization_build.training-operator.ids
  manifest = data.kustomization_build.training-operator.manifests[each.value]
}
