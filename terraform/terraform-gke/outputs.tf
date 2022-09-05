output "kubernetes_endpoint" {
  description = "The cluster endpoint"
  sensitive   = true
  value       = module.gke.endpoint
}

output "terraform_service_account" {
  description = "The default service account used for running nodes (user-provided)"
  value       = module.gke.service_account
}

output "cluster_identifier" {
  description = "Cluster name"
  value       = module.gke.name
}

output "private_network_identifier" {
  description = "The name of the VPC being created (user-provided)"
  value       = module.gcp-network.network_name
}

output "subnet_identifier" {
  description = "The name of the subnet being created (user-provided)"
  value       = module.gcp-network.subnets_names
}

output "subnet_ranges" {
  description = "The secondary ranges associated with the subnet (pods, and services)"
  value       = module.gcp-network.subnets_secondary_ranges
}
