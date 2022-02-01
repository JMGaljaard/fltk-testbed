from pathlib import Path

from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'dev'
    generate_docker(name)
    base_path = f'configs/{Path(__file__).parent.name}'
    exp_list = [
        'fedavg.yaml',
        'fednova.yaml',
        'fedprox.yaml',
        ]
    exp_list = [f'{base_path}/exps/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''

    print('Done')
