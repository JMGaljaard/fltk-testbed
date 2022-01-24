from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'dev'
    generate_docker(name, 10, True)
    base_path = 'configs/effect-freezing'
    exp_list = ['p_freezing-iid_freeze.yaml','p_freezing-iid_vanilla.yaml']
    exp_list = [f'{base_path}/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''

    print('Done')


