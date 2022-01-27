from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'p11_freezoff'
    generate_docker(name)
    base_path = 'configs/p15_freezoff_non_iid_large'
    exp_list = [
        # 'p15_freezoff_non_iid_offload_strict_large.yaml',
        # 'p15_freezoff_non_iid_offload_large.yaml',
        # 'p15_freezoff_non_iid_fedprox_large.yaml',
        # 'p15_freezoff_non_iid_fednova_large.yaml',
        'p15_freezoff_non_iid_dyn_terminate_swyh_large.yaml',
        # 'p15_freezoff_non_iid_fedavg_large.yaml',
        'p15_freezoff_non_iid_dyn_terminate_large.yaml',
        'p15_freezoff_non_iid_tifl_adaptive_large.yaml',
        'p15_freezoff_non_iid_tifl_basic_large.yaml'
        ]
    exp_list = [f'{base_path}/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''

    print('Done')




