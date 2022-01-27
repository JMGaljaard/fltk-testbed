from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'p11_freezoff'
    generate_docker(name)
    base_path = 'configs/p11_freezoff_iid'
    exp_list = [
        # 'p11_freezoff_iid_fedprox.yaml',
        # 'p11_freezoff_iid_fednova.yaml',
        # 'p11_freezoff_iid_offload.yaml',
        'p11_freezoff_iid_offload_strict.yaml',
        # 'p11_freezoff_iid_dyn_terminate_swyh.yaml',
        'p11_freezoff_iid_fedavg.yaml',
        'p11_freezoff_iid_tifl_adaptive.yaml',
        # 'p11_freezoff_iid_dyn_terminate.yaml',
        'p11_freezoff_iid_tifl_basic.yaml'
        ]
    exp_list = [f'{base_path}/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''

    print('Done')




