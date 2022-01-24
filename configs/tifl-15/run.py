from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'tifl-15'
    generate_docker(name)
    base_path = 'configs/tifl-15'
    exp_list = ['exp_p15_baseline.yaml',  'exp_p15_tifl-adaptive.yaml',  'exp_p15_tifl-basic.yaml']
    exp_list = [f'{base_path}/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''

    print('Done')


