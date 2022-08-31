data "google_client_config" "default" {}

# In case of using a gke deployment, use the GKE provider. Otherwise we create a container cluster.
module "gke" {
  source            = "terraform-google-modules/kubernetes-engine/google"
  project_id        = var.project_id
  name              = var.cluster_name
  # Create a ZONAL cluster, dissallowing the cluster to span multiple regions in a zone.
  # Alternatively, for scheduling cross-regions, utilize `zone` and `regions` instead of `regional` and `region`
  regional          = false
  region            = var.project_region
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


  node_pools                 = [
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

  node_pools_oauth_scopes = {
    all = []

    default-node-pool = [
      "https://www.googleapis.com/auth/cloud-platform",
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
    all = []

    default-node-pool = []

    medium-fltk-pool-1 = [
      {
        key    = "medium-fltk-pool-1"
        value  = true
        effect = "PREFER_NO_SCHEDULE"
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
