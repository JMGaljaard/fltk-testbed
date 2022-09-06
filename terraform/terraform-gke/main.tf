data "google_client_config" "default" {}

# Create a ZONAL cluster, disallowing the cluster to span multiple regions in a zone.
# Alternatively, for scheduling cross-regions, utilize `zone` and `regions` instead of `regional` and `region`
module "gke" {
  source            = "terraform-google-modules/kubernetes-engine/google"
  project_id        = var.project_id
  name              = var.cluster_name
  regional          = var.regional_deployment
  region            = var.project_region
  zones             = slice(var.project_zones, 0, 1)
  network           = module.gcp-network.network_name
  subnetwork        = module.gcp-network.subnets_names[0]
  ip_range_pods     = var.ip_range_pods_name
  ip_range_services = var.ip_range_services_name

  http_load_balancing        = false
  network_policy             = false
  horizontal_pod_autoscaling = false
  filestore_csi_driver       = false
  service_account            = local.terraform_service_account
  create_service_account     = false
  kubernetes_version         = var.kubernetes_version


  node_pools = [
    {
      name               = "default-node-pool"
      machine_type       = "e2-medium"
      node_locations     = "us-central1-c"
      auto_scaling       = false
      min_count          = 0
      max_count          = 1
      local_ssd_count    = 0
      spot               = false
      disk_size_gb       = 64
      disk_type          = "pd-standard"
      image_type         = "COS_CONTAINERD"
      enable_gcfs        = false
      enable_gvnic       = false
      auto_repair        = true
      auto_upgrade       = true
      service_account    = local.terraform_service_account
      preemptible        = false
      initial_node_count = 1
    },
    {
      name               = "medium-fltk-pool-1"
      machine_type       = "e2-medium"
      node_locations     = "us-central1-c"
      auto_scaling       = false
      min_count          = 0
      max_count          = 1
      local_ssd_count    = 0
      spot               = false
      disk_size_gb       = 64
      disk_type          = "pd-standard"
      image_type         = "COS_CONTAINERD"
      enable_gcfs        = false
      enable_gvnic       = false
      auto_repair        = true
      auto_upgrade       = true
      service_account    = local.terraform_service_account
      preemptible        = false
      initial_node_count = 0
    },
  ]
  # Allow the pods in the node pool to pull from gcr.io/<project-id>/<container-name>
  # Changes here after creation of a pool will not be updated by Terraform.
  node_pools_oauth_scopes = {
    all = [
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]
  }

  node_pools_labels = {
    all = {}

    default-node-pool = {
      default-node-pool = true
    }
  }

  node_pools_metadata = {
    all = {}

    default-node-pool = {}

    medium-fltk-pool-1 = {
      node-pool-metadata-custom-value = "medium-node-pool-fltk"
    }
  }

  node_pools_taints = {
    all               = []
    default-node-pool = [] # Default nodepool that will contain all the other pods

    # Taint node pool for scheduling testbed-pods only/preferentially
    medium-fltk-pool-1 = [
      {
        key    = "fltk.node"          # Taint is used in fltk pods
        value  = "medium-e2"          # In case more explicit matching is required
        effect = "PREFER_NO_SCHEDULE" # Other Pods are preferably not scheduled on this pool
      },
    ]
  }

  node_pools_tags = {
    all = []

    default-node-pool = [
      "default-node-pool",
    ]

    fltk-node-pool-1 = [
      "fltk-experiment-pool-1",
    ]
  }
}
