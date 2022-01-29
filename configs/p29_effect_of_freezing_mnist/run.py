from pathlib import Path

from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    EVENT_FILE="exp_events.txt"
    name = 'p28_non_iid_effect'
    generate_docker(name)
    base_path = f'configs/{Path(__file__).parent.name}'
    exp_list = [
        'fedavg-iid-freeze-0.yaml',
        'fedavg-iid-freeze-30.yaml',
        'fedavg-iid-freeze-70.yaml',
        'fedavg-non_iid-freeze-100.yaml',
        'fedavg-non_iid-freeze-40.yaml',
        'fedavg-non_iid-freeze-80.yaml',
        'fedavg-iid-freeze-100.yaml',
        'fedavg-iid-freeze-40.yaml',
        'fedavg-iid-freeze-80.yaml',
        'fedavg-non_iid-freeze-10.yaml',
        'fedavg-non_iid-freeze-50.yaml',
        'fedavg-non_iid-freeze-90.yaml',
        'fedavg-iid-freeze-10.yaml',
        'fedavg-iid-freeze-50.yaml',
        'fedavg-iid-freeze-90.yaml',
        'fedavg-non_iid-freeze-20.yaml',
        'fedavg-non_iid-freeze-60.yaml',
        'fedavg-iid-freeze-20.yaml',
        'fedavg-iid-freeze-60.yaml',
        'fedavg-non_iid-freeze-0.yaml',
        'fedavg-non_iid-freeze-30.yaml',
        'fedavg-non_iid-freeze-70.yaml',
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


