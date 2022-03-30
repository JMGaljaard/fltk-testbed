import argparse
import copy
from pathlib import Path
import yaml
import numpy as np


def load_yaml_file(file_path: Path):
    with open(file_path) as file:
        return yaml.full_load(file)


def generate_client(id, template: dict, world_size: int, type='default', cpu_set=None, num_cpus=1):
    local_template = copy.deepcopy(template)
    key_name = list(local_template.keys())[0]
    container_name = f'client_{type}_{id}'
    local_template[container_name] = local_template.pop(key_name)
    for key, item in enumerate(local_template[container_name]['environment']):
        if item == 'RANK={rank}':
            local_template[container_name]['environment'][key] = item.format(rank=id)
        if item == 'WORLD_SIZE={world_size}':
            local_template[container_name]['environment'][key] = item.format(world_size=world_size)
    local_template[container_name]['ports'] = [f'{5000+id}:5000']
    if cpu_set:
        local_template[container_name]['cpuset'] = f'{cpu_set}'
    else:
        local_template[container_name].pop('cpuset')
    local_template[container_name]['deploy']['resources']['limits']['cpus'] = f'{num_cpus}'
    return local_template, container_name


def gen_client(name: str, client_dict: dict, base_path: Path):
    """
    rank (id)
    num_cpu
    cpu_set
    name
    """
    client_descr_template = {
        'rank': 0,
        'num_cpu': 1,
        'num_cores': None,
        'name': name,
        'stub-file': 'stub.yml'
    }
    print(Path.cwd())
    mu = client_dict['cpu-speed']
    sigma = client_dict['cpu-variation']
    n = client_dict['amount']
    np.random.seed(0)
    stub_file = base_path / client_dict['stub-name']
    stub_data = load_yaml_file(stub_file)
    if client_dict['pin-cores'] is True:
        client_descr_template['num_cores'] = client_dict['num-cores']
    client_descr_template['stub-file'] = client_dict['stub-name']
    client_cpu_speeds = np.abs(np.round(np.random.normal(mu, sigma, size=n), 2))
    client_descriptions = []
    for cpu_speed in client_cpu_speeds:
        client_descr = copy.deepcopy(client_descr_template)
        client_descr['num_cpu'] = cpu_speed
        client_descriptions.append(client_descr)
    return client_descriptions


def generate_clients_proporties(clients_dict: dict, path: Path):
    results = []
    for k,v in clients_dict.items():
        results += gen_client(k, v, path)
    return results

def generate_compose_file_from_dict(system: dict):
    path = Path(system['base_path'])
    client_descriptions = generate_clients_proporties(system['clients'], path)
    last_core_id = 0
    world_size = len(client_descriptions) + 1
    system_template_path = path / 'system_stub.yml'

    system_template: dict = load_yaml_file(system_template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    if system['federator']['pin-cores']:
        cpu_set: str
        amount = system['federator']['num-cores']
        if amount > 1:
            cpu_set = f'{last_core_id}-{last_core_id + amount - 1}'
        else:
            cpu_set = f'{last_core_id}'
        system_template['services']['fl_server']['cpuset'] = cpu_set
        last_core_id += amount
    else:
        system_template['services']['fl_server'].pop('cpuset')
    for idx, client_d in enumerate(client_descriptions):
        stub_file = path / client_d['stub-file']
        stub_data = load_yaml_file(stub_file)
        cpu_set = None
        if client_d['num_cores']:
            amount = client_d['num_cores']
            if amount > 1:
                cpu_set = f'{last_core_id}-{last_core_id + amount - 1}'
            else:
                cpu_set = f'{last_core_id}'
            last_core_id += amount
        local_template, container_name = generate_client(idx + 1, stub_data, world_size, client_d['name'], cpu_set,
                                                         client_d['num_cpu'])
        system_template['services'].update(local_template)
        print(container_name)
    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_compose_file(path: Path):
    """
    Used properties:
    - World size
    - num clients?
    - path to deploy files
    - random seed?
    """

    system_path = path / 'description.yml'
    system = load_yaml_file(system_path)
    # path = Path('deploy/dev_generate')
    generate_compose_file_from_dict(system)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate docker-compose file')
    parser.add_argument('path', type=str,
                        help='Path to a deployment config folder')
    parser.add_argument('--clients', type=int, help='Set the number of clients in the system', default=None)
    args = parser.parse_args()
    path = Path(args.path)
    results = generate_compose_file(path)
    print('done')