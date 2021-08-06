import sys
import yaml
import copy

template_path = './deploy/templates'

def load_system_template():
    with open(f'{template_path}/system_stub.yml') as file:
        documents = yaml.full_load(file)
        return documents

def load_client_template(type='default'):
    with open(f'{template_path}/client_stub_{type}.yml') as file:
        documents = yaml.full_load(file)
        return documents

def generate_client(id, template: dict, world_size: int, type='default'):
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
    return local_template, container_name


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


if __name__ == '__main__':

    num_clients = int(sys.argv[1])
    generate(num_clients)
    print('Done')

