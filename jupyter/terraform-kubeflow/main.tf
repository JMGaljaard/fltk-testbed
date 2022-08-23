
data "google_client_config" "default" {}

data "google_container_cluster" "testbed_cluster" {
  name     = var.cluster_name
  location = var.project_region
}

# Setup kubeflow with kubeflow "kustomization" provided by WOGRA-AG
#module "kubeflow" {
#  source = "WOGRA-AG/kubeflow/kustomization"
#
#  dex_user_email = "my@example.com"
#
#  # Disable 'production' features of Kubeflow. Change if need, e.g. "serving" for inference.
#  deploy_notebooks = false
#  deploy_dashboard = false
#  deploy_katib = false
#  deploy_tensorboard = false
#  deploy_volumes = false
#  deploy_serving = false
#}


data "kustomization_build" "training-operator" {
  path = "github.com/kubeflow/manifests.git/apps/training-operator/upstream/overlays/standalone?ref=${var.kubeflow_version}"
}

resource "kustomization_resource" "training-operator" {
  for_each = data.kustomization_build.training-operator.ids
  manifest = data.kustomization_build.training-operator.manifests[each.value]

}