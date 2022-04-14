import copy
from pathlib import Path
import os
import yaml
from fltk.util.generate_docker_compose_2 import generate_compose_file_from_dict


def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        # else:
        #     rm_tree(child)
    # pth.rmdir()


def check_num_clients_consistency(cfg_data: dict):
    if type(cfg_data) is str:
        cfg_data = yaml.safe_load(copy.deepcopy(cfg_data))

    if 'deploy' in cfg_data and 'docker' in cfg_data['deploy']:
        num_docker_clients = sum([x['amount'] for x in cfg_data['deploy']['docker']['clients'].values()])
        if cfg_data['num_clients'] != num_docker_clients:
            print('[Warning]\t Number of docker clients is not equal to the num_clients property!')


def generate(base_path: Path, config_path: Path, **kwargs):
    descr_path = base_path / 'descr.yaml'

    exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
    descr_data = ''
    with open(descr_path) as descr_f:
        descr_data = descr_f.read()
    exps_path = base_path / 'exps'
    rm_tree(exps_path)
    exps_path.mkdir(parents=True, exist_ok=True)

    check_num_clients_consistency(descr_data)
    for exp_cfg in exp_cfg_list:
        exp_cfg_data = ''
        with open(exp_cfg) as exp_f:
            exp_cfg_data = exp_f.read()

        exp_data = descr_data + exp_cfg_data
        exp_data += f'\nexperiment_prefix: \'{base_path.name}_{exp_cfg.name.split(".")[0]}\'\n'
        filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
        with open(exps_path / filename, mode='w') as f:
            f.write(exp_data)
    print('Done')


# def run():
#     base_path = Path(__file__).parent
#     descr_path = base_path / 'descr.yaml'
#
#     exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
#     descr_data = ''
#     with open(descr_path) as descr_f:
#         descr_data = descr_f.read()
#
#     exps_path = base_path / 'exps'
#     exps_path.mkdir(parents=True, exist_ok=True)
#     for exp_cfg in exp_cfg_list:
#         exp_cfg_data = ''
#         replications = 1
#         with open(exp_cfg) as exp_f:
#             exp_cfg_data = exp_f.read()
#         for replication_id in range(replications):
#             exp_data = descr_data + exp_cfg_data
#             exp_data += f'\nexperiment_prefix: \'{Path(__file__).parent.name}_{exp_cfg.name.split(".")[0]}\'\n'
#             filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
#             with open(exps_path / filename, mode='w') as f:
#                 f.write(exp_data)
#     print('Done')


def run(base_path: Path, config_path: Path, **kwargs):
    print(f'Run {base_path}')
    print(list(base_path.iterdir()))
    descr_path = base_path / 'descr.yaml'
    exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
    descr_data = ''
    with open(descr_path) as descr_f:
        descr_data = yaml.safe_load(descr_f.read())

    replications = 1
    if 'replications' in descr_data:
        replications = descr_data['replications']
    run_docker = False
    if 'deploy' in descr_data and 'docker' in descr_data['deploy']:
    # if 'docker_system' in descr_data:
        # Run in docker
        # Generate Docker
        print(descr_data)
        docker_deploy_path = Path(descr_data['deploy']['docker']['base_path'])

        print(docker_deploy_path)
        run_docker = True
        generate_compose_file_from_dict(descr_data['deploy']['docker'])
        # generate_compose_file(docker_deploy_path)

    exp_files = [x for x in (base_path / 'exps').iterdir() if x.suffix in ['.yaml', '.yml']]

    cmd_list = []
    print(exp_files)
    if run_docker:
        first_prefix = '--build'
        for exp_cfg_file in exp_files:
            for replication_id in range(replications):
                cmd = f'export OPTIONAL_PARAMS="--prefix={replication_id}";export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
                cmd_list.append(cmd)
                # print(f'Running cmd: "{cmd}"')
                # os.system(cmd)
                first_prefix = ''
    else:
        print('Switching to direct mode')
        for exp_cfg_file in exp_files:
            for replication_id in range(replications):
                # cmd = f'export OPTIONAL_PARAMS="--prefix={replication_id}";export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
                cmd = f'python3 -m fltk single {exp_cfg_file} --prefix={replication_id}'
                cmd_list.append(cmd)

    for cmd in cmd_list:
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
    print('Done')
    # docker_system


    # name = 'dev'
    # generate_docker(name)
    # base_path = f'{Path(__file__).parent}'
    # exp_list = [
    #     'fedavg.yaml',
    #     ]
    # exp_list = [f'{base_path}/exps/{x}' for x in exp_list]
    # first_prefix = '--build'
    # for exp_cfg_file in exp_list:
    #     cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
    #     print(f'Running cmd: "{cmd}"')
    #     os.system(cmd)
    #     first_prefix = ''

    # print('Done')

# if __name__ == '__main__':
#     base_path = Path(__file__).parent
#     descr_path = base_path / 'descr.yaml'
#
#     exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
#     descr_data = ''
#     with open(descr_path) as descr_f:
#         descr_data = descr_f.read()
#     exps_path = base_path / 'exps'
#     exps_path.mkdir(parents=True, exist_ok=True)
#     for exp_cfg in exp_cfg_list:
#         exp_cfg_data = ''
#         with open(exp_cfg) as exp_f:
#             exp_cfg_data = exp_f.read()
#
#         exp_data = descr_data + exp_cfg_data
#         exp_data += f'\nexperiment_prefix: \'{Path(__file__).parent.name}_{exp_cfg.name.split(".")[0]}\'\n'
#         filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
#         with open(exps_path / filename, mode='w') as f:
#             f.write(exp_data)
#     print('Done')
#
#
