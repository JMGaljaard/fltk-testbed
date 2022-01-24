from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'terminate'
    generate_docker(name, 16, True)
    base_path = 'configs/terminate'
    exp_list = ['p_terminate_terminate_swyh.yaml', 'p_terminate_terminate.yaml', 'p_terminate_vanilla.yaml']
    exp_list = [f'{base_path}/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''

    print('Done')


