import sys
import yaml
import copy
import argparse

global_template_path = './deploy/templates'

def load_system_template(template_path = global_template_path):
    print(f'Loading system template from {template_path}/system_stub.yml')
    with open(f'{template_path}/system_stub.yml') as file:
        documents = yaml.full_load(file)
        return documents

def load_client_template(type='default', template_path = global_template_path):
    with open(f'{template_path}/client_stub_{type}.yml') as file:
        documents = yaml.full_load(file)
        return documents

def get_deploy_path(name: str):
    return f'./deploy/{name}'


def generate_client(id, template: dict, world_size: int, type='default', cpu_set=''):
    local_template = copy.deepcopy(template)
    key_name = list(local_template.keys())[0]
    container_name = f'client_{type}_{id}'
    local_template[container_name] = local_template.pop(key_name)
    for key, item in enumerate(local_template[container_name]['environment']):
        if item == 'RANK={rank}':
            local_template[container_name]['environment'][key] = item.format(rank=id)
        if item == 'WORLD_SIZE={world_size}':
            local_template[container_name]['environment'][key] = item.format(world_size=world_size)
    # for key, item in enumerate(local_template[container_name]):
    #     if item == 'cpuset: {cpu_set}':
    #         local_template[container_name][key] = item.format(cpu_set=cpu_set)

    local_template[container_name]['ports'] = [f'{5000+id}:5000']
    local_template[container_name]['cpuset'] = f'{cpu_set}'
    return local_template, container_name

def generate_compose_file():
    print()


def generate_p30_freezing_effect_dev():
    template_path = get_deploy_path('p28_non_iid_effect')
    num_clients = 6
    cpu_per_client = 1
    num_cpus = 20
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_set = 0
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        cpu_set = f'{cpu_idx}'
        cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type, cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_p28_non_iid_effect():
    template_path = get_deploy_path('p28_non_iid_effect')
    num_clients = 10
    cpu_per_client = 1
    num_cpus = 20
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_set = 0
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        cpu_set = f'{cpu_idx}'
        cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type, cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_p23_freezoff_w9s3():
    template_path = get_deploy_path('p23_freezoff_w9s3')
    num_clients = 9
    cpu_per_client = 1
    num_cpus = 20
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_set = 0
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        if 0 < client_id <= 3:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 3 < client_id <= 6:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}-{cpu_idx+1}'
            cpu_idx += 2
        elif 6 < client_id <= 9:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}-{cpu_idx + 2}'
            cpu_idx += 3
        else:
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type, cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_p23_freezoff_w9s3_half():
    template_path = get_deploy_path('p23_freezoff_w9s3-half')
    num_clients = 9
    cpu_per_client = 1
    num_cpus = 20
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_set = 0
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        if 0 < client_id <= 3:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 3 < client_id <= 6:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}-{cpu_idx+1}'
            cpu_idx += 2
        elif 6 < client_id <= 9:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}-{cpu_idx + 2}'
            cpu_idx += 3
        else:
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type, cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)



def generate_terminate(num_clients = 16, medium=False):
    template_path = get_deploy_path('terminate')
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        if client_id < 5:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        else:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_dev(num_clients = 2, medium=False):
    template_path = get_deploy_path('dev')
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        if not medium:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}-{cpu_idx + 2}'
            cpu_idx += 3
        else:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_p13_w6():
    template_path = get_deploy_path('p11_freezoff')
    num_clients= 6
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        if 0 < client_id <= 2:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 2 < client_id <= 4:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 4 < client_id <= 6:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        else:
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)


def generate_check_w4():
    template_path = get_deploy_path('p11_freezoff')
    num_clients= 4
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'fast'
        cpu_set = f'{cpu_idx}'
        cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_check_w18():
    template_path = get_deploy_path('p11_freezoff')
    num_clients= 18
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'fast'
        cpu_set = f'{cpu_idx}'
        cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)


def generate_p11_freezoff():
    template_path = get_deploy_path('p11_freezoff')
    num_clients= 18
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        if 0 < client_id <= 6:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 6 < client_id <= 12:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 12 < client_id <= 18:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        else:
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)


def generate_tifl_15():
    template_path = get_deploy_path('tifl-15')
    num_clients= 18
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 2
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        if 0 < client_id <= 6:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 6 < client_id <= 12:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 12 < client_id <= 18:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        else:
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)


def generate_tifl_3():
    template_path = get_deploy_path('tifl-15')
    num_clients= 3
    world_size = num_clients + 1
    system_template: dict = load_system_template(template_path=template_path)

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_idx = 3
    for client_id in range(1, num_clients + 1):
        client_type = 'default'
        if 0 < client_id <= 1:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 1 < client_id <= 2:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif 2 < client_id <= 3:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        else:
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1

        client_template: dict = load_client_template(type=client_type, template_path=template_path)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type,
                                                            cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate_offload_exp():
    num_clients = 4
    cpu_per_client = 1
    num_cpus = 20
    world_size = num_clients + 1
    system_template: dict = load_system_template()

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)
    cpu_set = 0
    cpu_idx = 3
    for client_id in range(1, num_clients + 1):
        client_type = 'medium'
        client_type = 'default'
        if client_id == 1 or client_id == 2:
            client_type = 'medium'
            cpu_set = f'{cpu_idx}-{cpu_idx+1}'
            cpu_idx += 2
        elif client_id == 3:
            client_type = 'slow'
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1
        elif client_id == 4:
            client_type = 'fast'
            cpu_set = f'{cpu_idx}-{cpu_idx + 2}'
            cpu_idx += 3
        else:
            cpu_set = f'{cpu_idx}'
            cpu_idx += 1

        client_template: dict = load_client_template(type=client_type)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type, cpu_set=cpu_set)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def generate(num_clients: int):
    world_size = num_clients + 1
    system_template :dict = load_system_template()

    for key, item in enumerate(system_template['services']['fl_server']['environment']):
        if item == 'WORLD_SIZE={world_size}':
            system_template['services']['fl_server']['environment'][key] = item.format(world_size=world_size)

    for client_id in range(1, num_clients+1):
        client_type = 'default'
        if client_id == 1:
            client_type='slow'
        if client_id == 2:
            client_type='medium'
        client_template: dict = load_client_template(type=client_type)
        client_definition, container_name = generate_client(client_id, client_template, world_size, type=client_type)
        system_template['services'].update(client_definition)

    with open(r'./docker-compose.yml', 'w') as file:
        yaml.dump(system_template, file, sort_keys=False)

def run(name, num_clients = None, medium=False):
    exp_dict = {
        'tifl-15': generate_tifl_15,
        'dev': generate_dev,
        'terminate': generate_terminate,
        'p11_freezoff': generate_p11_freezoff,
        'p13_w6' : generate_p13_w6,
        'p23_w9s3': generate_p23_freezoff_w9s3,
        'p23_w9s3-half': generate_p23_freezoff_w9s3_half,
        'p28_non_iid_effect': generate_p28_non_iid_effect,
        'p30_dev': generate_p30_freezing_effect_dev,
        'generate_check_w4': generate_check_w4,
        'generate_check_w18': generate_check_w18
    }
    if num_clients:
        exp_dict[name](num_clients, medium)
    else:
        exp_dict[name]()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate docker-compose file')
    parser.add_argument('name', type=str,
                        help='Name of an experiment')
    parser.add_argument('--clients', type=int, help='Set the number of clients in the system', default=None)
    args = parser.parse_args()
    run(args.name, args.clients)
    print('Done')

