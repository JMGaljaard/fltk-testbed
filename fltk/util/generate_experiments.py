from pathlib import Path


def generate(base_path: Path):
    descr_path = base_path / 'descr.yaml'

    exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
    descr_data = ''
    with open(descr_path) as descr_f:
        descr_data = descr_f.read()
    exps_path = base_path / 'exps'
    exps_path.mkdir(parents=True, exist_ok=True)
    for exp_cfg in exp_cfg_list:
        exp_cfg_data = ''
        with open(exp_cfg) as exp_f:
            exp_cfg_data = exp_f.read()

        exp_data = descr_data + exp_cfg_data
        exp_data += f'\nexperiment_prefix: \'{Path(__file__).parent.name}_{exp_cfg.name.split(".")[0]}\'\n'
        filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
        with open(exps_path / filename, mode='w') as f:
            f.write(exp_data)
    print('Done')

# if __name__ == '__main__':
#     base_path = Path(__file__).parent
#     descr_path = base_path / 'descr.yaml'
#
#     exp_cfg_list = [x for x in base_path.iterdir() if '.cfg' in x.suffixes]
#     descr_data = ''
#     with open(descr_path) as descr_f:
#         descr_data = descr_f.read()
#     exps_path = base_path / 'exps'
#     exps_path.mkdir(parents=True, exist_ok=True)
#     for exp_cfg in exp_cfg_list:
#         exp_cfg_data = ''
#         with open(exp_cfg) as exp_f:
#             exp_cfg_data = exp_f.read()
#
#         exp_data = descr_data + exp_cfg_data
#         exp_data += f'\nexperiment_prefix: \'{Path(__file__).parent.name}_{exp_cfg.name.split(".")[0]}\'\n'
#         filename = '.'.join([exp_cfg.name.split('.')[0], exp_cfg.name.split('.')[2]])
#         with open(exps_path / filename, mode='w') as f:
#             f.write(exp_data)
#     print('Done')
#
#
