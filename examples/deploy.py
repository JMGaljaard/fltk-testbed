import argparse
import os
import time
import googleapiclient.discovery
from googleapiclient.errors import HttpError


def create_federator(compute, project, zone, name_template, rank, region, machine_image):
    machine_type = f'zones/{zone}/machineTypes/g1-small'
    instance_name = name_template.format(rank=rank)
    subnetwork = f'projects/{project}/regions/{region}/subnetworks/default'

    print(instance_name)
    client_config = {
        "kind": "compute#instance",
        "name": instance_name,
        "zone": zone,
        "minCpuPlatform": "Automatic",
        "machineType": machine_type,
        "displayDevice": {
            "enableDisplay": False
        },
        "metadata": {
            "kind": "compute#metadata",
            "items": [],
        },
        "tags": {
            "items": [
                "http-server",
                "https-server"
            ]
        },
        "canIpForward": False,
         "networkInterfaces": [
            {
              "kind": "compute#networkInterface",
              "subnetwork": subnetwork,
              "accessConfigs": [
                {
                  "kind": "compute#accessConfig",
                  "name": "External NAT",
                  "type": "ONE_TO_ONE_NAT",
                  "networkTier": "PREMIUM"
                }
              ],
              "aliasIpRanges": []
            }
          ],
        "description": "",
        "labels": {
            "experiment": "ex-c20"
        },
        "scheduling": {
            "preemptible": False,
            "onHostMaintenance": "MIGRATE",
            "automaticRestart": True,
            "nodeAffinities": []
        },
        "deletionProtection": False,
        "reservationAffinity": {
            "consumeReservationType": "ANY_RESERVATION"
        },
        "serviceAccounts": [
            {
                "email": "default",
                "scopes": [
                    "https://www.googleapis.com/auth/devstorage.read_only",
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring.write",
                    "https://www.googleapis.com/auth/servicecontrol",
                    "https://www.googleapis.com/auth/service.management.readonly",
                    "https://www.googleapis.com/auth/trace.append"
                ]
            }
        ],
        "sourceMachineImage": machine_image,
        "shieldedInstanceConfig": {
            "enableSecureBoot": False,
            "enableVtpm": False,
            "enableIntegrityMonitoring": True
        },
        "confidentialInstanceConfig": {
            "enableConfidentialCompute": False
        }
    }
    return compute.instances().insert(
        project=project,
        zone=zone,
        body=client_config).execute()

def create_client(compute, project, zone, name_template, rank, world_size, host, nic, region, machine_image):
    machine_type = f'zones/{zone}/machineTypes/g1-small'
    instance_name = name_template.format(rank=rank)
    subnetwork = f'projects/{project}/regions/{region}/subnetworks/default'
    startup_script = open(
        os.path.join(
            os.path.dirname(__file__), 'startup-script_template.sh'), 'r').read()
    startup_args = {
        'rank_arg': rank,
        'world_size_arg': world_size,
        'host_arg': host,
        'nic_arg': nic
    }
    startup_script = startup_script.format(**startup_args)
    print(instance_name)

    client_config = {
        "kind": "compute#instance",
        "name": instance_name,
        "zone": zone,
        "minCpuPlatform": "Automatic",
        "machineType": machine_type,
        "displayDevice": {
            "enableDisplay": False
        },
        "metadata": {
            "kind": "compute#metadata",
            "items": [
                {
                # Startup script is automatically executed by the
                # instance upon startup.
                'key': 'startup-script',
                'value': startup_script
            }
            ],
        },
        "tags": {
            "items": []
        },
        "canIpForward": False,
        "networkInterfaces": [
            {
                "kind": "compute#networkInterface",
                "subnetwork": subnetwork,
                "aliasIpRanges": []
            }
        ],
        "description": "",
        "labels": {
            "experiment": "ex-c20"
        },
        "scheduling": {
            "preemptible": False,
            "onHostMaintenance": "MIGRATE",
            "automaticRestart": True,
            "nodeAffinities": []
        },
        "deletionProtection": False,
        "reservationAffinity": {
            "consumeReservationType": "ANY_RESERVATION"
        },
        "serviceAccounts": [
            {
                "email": "default",
                "scopes": [
                    "https://www.googleapis.com/auth/devstorage.read_only",
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring.write",
                    "https://www.googleapis.com/auth/servicecontrol",
                    "https://www.googleapis.com/auth/service.management.readonly",
                    "https://www.googleapis.com/auth/trace.append"
                ]
            }
        ],
        "sourceMachineImage": machine_image,
        "shieldedInstanceConfig": {
            "enableSecureBoot": False,
            "enableVtpm": False,
            "enableIntegrityMonitoring": True
        },
        "confidentialInstanceConfig": {
            "enableConfidentialCompute": False
        }
    }
    return compute.instances().insert(
        project=project,
        zone=zone,
        body=client_config).execute()

# [START list_instances]
def list_instances(compute, project, zone):
    result = compute.instances().list(project=project, zone=zone).execute()

    result2 = compute.machineImages().list(project=project).execute()
    print(result2)
    return result['items'] if 'items' in result else None
# [END list_instances]

# [START wait_for_operation]
def wait_for_operation(compute, project, zone, operation):
    print('Waiting for operation to finish...')
    while True:
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation).execute()

        if result['status'] == 'DONE':
            print("done.")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(1)
# [END wait_for_operation]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create VMs in GCP for Federated Learning')
    parser.add_argument('--num_clients', type=int, default=20,
                        help='The number of clients (excluding the Federator) in the system')
    parser.add_argument('--project', type=str, default='tud-federated-learning',
                        help='The Google Cloud Platform project name')
    parser.add_argument('--machine_image', type=str, default='c20-machine-image',
                        help='The Google Cloud Platform project name')
    args = parser.parse_args()

    num_clients = args.num_clients
    project_name = args.project
    machine_image_name = args.machine_image

    # Change these values if desired
    region = 'europe-west4'
    zone_name = f'{region}-a'
    instance_name='tud-federated-learning-automated-instance'
    name_template = 'tud-fl-client-{rank}'
    name_template_federator = 'tud-fl-federator-{rank}'
    world_size = num_clients + 1
    nic = 'ens4' # Default nic in GCP ubuntu machines
    machine_image = f'projects/{project_name}/global/machineImages/{machine_image_name}'
    compute = googleapiclient.discovery.build('compute', 'beta')

    ######################
    ## Create Federator ##
    ######################
    try:
        federator_operation = create_federator(compute, project_name, zone_name, name_template_federator, 0, region, machine_image)
        wait_for_operation(compute, project_name, zone_name, federator_operation['name'])
    except HttpError as http_error:
        if http_error.status_code == 409 and http_error.error_details[0]['reason'] == 'alreadyExists':
            print('Resource already exists, continue with the next')
        else:
            raise http_error

    instances = list_instances(compute, project_name, zone_name)
    federator_ip = [x['networkInterfaces'][0]['networkIP'] for x in instances if x['name']==name_template_federator.format(rank=0)][0]
    host = federator_ip

    ####################
    ## Create Clients ##
    ####################
    operations = []
    for id in range(1, num_clients+1):
        try:
            operations.append(create_client(compute, project_name, zone_name, name_template, id, world_size, host, nic, region, machine_image))
            wait_for_operation(compute, project_name, zone_name, operations[-1]['name'])
        except HttpError as http_error:
            if http_error.status_code == 409 and http_error.error_details[0]['reason'] == 'alreadyExists':
                print('Resource already exists, continue with the next')
                continue
            else:
                raise http_error
    for operation in operations:
        wait_for_operation(compute, project_name, zone_name, operation['name'])

    print("""Now login via ssh into the federator VM and start the experiment.""")

