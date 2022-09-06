
data "google_client_config" "default" {}

# Retrieve kustomize templates
data "kustomization_build" "training_operator" {
  path = "github.com/kubeflow/manifests.git/apps/training-operator/upstream/overlays/standalone?ref=${var.kubeflow_version}"
}

# Deploy resources one-by-one.
resource "kustomization_resource" "training_operator" {
  for_each = data.kustomization_build.training_operator.ids
  manifest = data.kustomization_build.training_operator.manifests[each.value]
}

# Create NFS resource
resource "helm_release" "nfs_client_provisioner" {
  name       = var.nfs_provider_information.release_name
  repository = var.nfs_provisioner_repo_url
  chart      = var.nfs_provider_information.chart_name

  namespace        = var.nfs_provider_information.namespace
  create_namespace = true

  values = [
    templatefile("${path.module}/values.nfs.yaml.tpl", {
      nfs_server_path  = var.nfs_provider_information.server_path
      image_repository = var.nfs_provider_information.image_repository
      image_tag        = var.nfs_provider_information.image_tag
      pull_policy      = var.nfs_provider_information.pull_policy
      nfs_size         = var.nfs_provider_information.storage_size
    })
  ]
}
