from pathlib import Path

if __name__ == '__main__':
    base_path = 'configs/dev'

    path = Path(base_path)
    descr_path = path / 'descr.yaml'

    exp_cfg_list = [x for x in path.iterdir() if '.cfg' in x.suffixes]
    descr_data = ''
    with open(descr_path) as descr_f:
        descr_data = descr_f.read()
    exps_path = path / 'exps'
    exps_path.mkdir(parents=True, exist_ok=True)
    for exp_cfg in exp_cfg_list:
        exp_cfg_data = ''
        with open(exp_cfg) as exp_f:
            exp_cfg_data = exp_f.read()

        exp_data = descr_data + exp_cfg_data
        exp_data += f'\nexperiment_prefix: \'{exp_cfg.name.split(".")[0]}\'\n'
        filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
        with open(exps_path / filename, mode='w') as f:
            f.write(exp_data)
    print('Done')


