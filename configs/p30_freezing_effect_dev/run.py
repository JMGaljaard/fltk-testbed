from pathlib import Path

from fltk.util.generate_docker_compose import run as generate_docker
import os
if __name__ == '__main__':
    name = 'p30_dev'
    generate_docker(name)
    base_path = f'configs/{Path(__file__).parent.name}'
    exp_list = [
        'fedavg-iid-freeze-0.yaml',
        'fedavg-iid-freeze-16.yaml',
        'fedavg-iid-freeze-50.yaml',
        'fedavg-iid-freeze-83.yaml',
        'fedavg-non-iid-freeze-100.yaml',
        'fedavg-non-iid-freeze-33.yaml',
        'fedavg-non-iid-freeze-66.yaml',
        'fedavg-iid-freeze-100.yaml',
        'fedavg-iid-freeze-33.yaml',
        'fedavg-iid-freeze-66.yaml',
        'fedavg-non-iid-freeze-0.yaml',
        'fedavg-non-iid-freeze-16.yaml',
        'fedavg-non-iid-freeze-50.yaml',
        'fedavg-non-iid-freeze-83.yaml',
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


