from pathlib import Path
import time
from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    EVENT_FILE="exp_events.txt"
    name = 'p23_w9s3_fast'
    # name = 'p23_w9s3'
    generate_docker(name)
    base_path = f'configs/{Path(__file__).parent.name}'
    exp_list = [
        'fedavg.yaml',
        # 'offload_strict.yaml',
        # 'offload_strict2.yaml',
        # 'offload_strict3.yaml',
        # 'offload_strict4.yaml',
        # 'fednova.yaml',
        # 'fedprox.yaml',
        # 'tifl_adaptive.yaml',
        # 'tifl_basic.yaml',
        # 'offload.yaml',
        # 'dyn_terminate_swyh.yaml',
        # 'dyn_terminate.yaml',
        ]
    exp_list = [f'{base_path}/exps/{x}' for x in exp_list]
    first_prefix = '--build'
    for exp_cfg_file in exp_list:
        cmd = f'export EXP_CONFIG_FILE="{exp_cfg_file}"; docker-compose --compatibility up {first_prefix};'
        os.system(f'echo "[$(date +"%T")] Starting {exp_cfg_file}" >> {EVENT_FILE}')
        start = time.time()


        print(f'Running cmd: "{cmd}"')
        os.system(cmd)
        first_prefix = ''
        elapsed = time.time() - start
        os.system(f'echo "[$(date +"%T")] Finished with {exp_cfg_file} in {elapsed} seconds" >> {EVENT_FILE}')

    print('Done')


