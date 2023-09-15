data "google_client_config" "default" {}

# Add Vulcano Gang scheduler plugin using all default values.
resource "helm_release" "volcano_scheduler" {
  name       = var.vulcano_scheduler_information.release_name
  repository = var.vulcano_scheduler_repo_url
  chart      = var.vulcano_scheduler_information.chart_name
  version    = var.vulcano_scheduler_information.version

  namespace        = var.vulcano_scheduler_information.namespace
  create_namespace = true
}

# Treat training-operator as overlay and apply a patch to add support for gang scheduling.
# Creates an overlay (patched version) of the original training operator to deploy.
data "kustomization_overlay" "training_operator" {
  resources = [
    "github.com/kubeflow/manifests.git/apps/training-operator/upstream/overlays/standalone?ref=${var.kubeflow_version}"
  ]

  # Apply vulcano patch in overlay.
  patches {
    path = "patches/training-operator-patch.yaml"
    target {
      kind      = "Deployment"
      namespace = "kubeflow"
      name      = "training-operator"
    }
  }
}

# Deploy resources one-by-one.
resource "kustomization_resource" "training_operator" {
  # Before we can install the training operator, we need to have the vulcano_scheduler up and running.
  # See also the patch that we apply to the training operator through kustomize.
  depends_on = [helm_release.volcano_scheduler]
  for_each   = data.kustomization_overlay.training_operator.ids
  manifest   = data.kustomization_overlay.training_operator.manifests[each.value]
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
