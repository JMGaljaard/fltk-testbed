from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'p13_w6'
    generate_docker(name)
    base_path = 'configs/p13_variance_dev'
    exp_list = [
        'p13_variance_dev_offload_large.yaml',
        # 'p13_variance_dev_fedprox_large.yaml',
        # 'p13_variance_dev_fednova_large.yaml',
        # 'p13_variance_dev_dyn_terminate_swyh_large.yaml',
        # 'p13_variance_dev_fedavg_large.yaml',
        # 'p13_variance_dev_tifl_adaptive_large.yaml',
        # 'p13_variance_dev_dyn_terminate_large.yaml',
        # 'p13_variance_dev_tifl_basic_large.yaml'
        ]
    exp_list = [f'{base_path}/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''

    print('Done')




