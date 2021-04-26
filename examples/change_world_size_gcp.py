import argparse
import os
import time
import googleapiclient.discovery
from googleapiclient.errors import HttpError

def update_startup_script(compute, project, zone, name_template, rank, world_size, host, nic, region):
    instance_name = name_template.format(rank=rank)
    startup_script = open(
        os.path.join(
            os.path.dirname(__file__), 'startup-script_template.sh'), 'r').read()
    startup_args = {
        'rank_arg': rank,
        'world_size_arg': world_size,
        'host_arg': host,
        'nic_arg': nic
    }
    instanceget = compute.instances().get(project=project, zone=zone, instance=instance_name).execute()

    fingerprint = instanceget['metadata']['fingerprint']
    instance_id = instanceget['id']
    # Insert values for startup script in template
    startup_script = startup_script.format(**startup_args)
    client_body = {
        "fingerprint": fingerprint,
        "items": [
            {
                "key": "startup-script",
                "value": startup_script
            }
        ]
    }
    print(f'Changing startup script of instance {instance_name}')
    return compute.instances().setMetadata(
        project=project,
        zone=zone,
        instance=instance_id,
        body=client_body).execute()

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

    parser = argparse.ArgumentParser(description='Change the world-size of VMs in GCP')
    parser.add_argument('--num_clients', type=int, default=20, help='The number of clients (excluding the Federator) in the system')
    parser.add_argument('--project', type=str, default='tud-federated-learning', help='The Google Cloud Platform project name')
    args = parser.parse_args()

    num_clients = args.num_clients
    project_name = args.project

    # Change these values if desired
    region = 'europe-west4'
    zone_name = f'{region}-a'
    instance_name='tud-federated-learning-automated-instance'
    name_template = 'tud-fl-client-{rank}'
    name_template_federator = 'tud-fl-federator-{rank}'

    # The world size is number of clients + 1
    world_size = num_clients + 1
    nic = 'ens4' # Default nic in GCP ubuntu machines

    # Create GCP API instance
    compute = googleapiclient.discovery.build('compute', 'beta')
    instances = list_instances(compute, project_name, zone_name)
    federator_ip = [x['networkInterfaces'][0]['networkIP'] for x in instances if x['name']==name_template_federator.format(rank=0)][0]
    host = federator_ip

    ############################
    ## Alter Clients metadata ##
    ############################
    operations = []
    for id in range(1, num_clients+1):
        try:
            operations.append(update_startup_script(compute, project_name, zone_name, name_template, id, world_size, host, nic, region))
        except HttpError as http_error:
            if http_error.status_code == 409 and http_error.error_details[0]['reason'] == 'alreadyExists':
                print('Resource already exists, continue with the next')
                continue
            else:
                raise http_error
    for operation in operations:
        wait_for_operation(compute, project_name, zone_name, operation['name'])

    print("""The world-size of the clients are updated""")

