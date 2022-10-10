import jinja2
from functools import reduce

NUMBER_OF_PARAMETERS = 6


def generate_binary_strings(bit_count):
    """Taken from:
    https://stackoverflow.com/questions/64890117/what-is-the-best-way-to-generate-all-binary-strings-of-the-given-length-in-pytho
    """

    binary_strings = []

    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')

    genbin(bit_count)

    return binary_strings


binary_strings = generate_binary_strings(NUMBER_OF_PARAMETERS)
split_binary_strings = list(map(lambda xs: [*xs], binary_strings))

experiment_indices, _ = list(
    zip(*filter(
        lambda x: x[1] == 1,
        enumerate(
            map(
                lambda xs: reduce(lambda x, y: x * y, xs),
                map(lambda xs: map(lambda x: 2 * int(x) - 1, xs),
                    split_binary_strings))))))

cores = ["500m", "2000m"]
memories = ["1Gi", "8Gi"]
train_batches = [32, 256]
test_batches = [32, 256]
parallel_list = [2, 50]
networks = [
    '{ "network": "Cifar10CNN", "lossFunction": "CrossEntropyLoss", "dataset": "cifar10" }',
    '{ "network": "ResNet34", "lossFunction": "CrossEntropyLoss", "dataset": "cifar10" }'
]
seeds = [42, 360, 20]

parameters = [
    cores, memories, train_batches, test_batches, parallel_list, networks
]

# Based on the indices of the experiments, we can now generate the experiments knowing that we should take either the last entry of each
# list or the first entry of each list depending on whether the index is 0 or 1.

experiments = []

for on_off_values in map(
        lambda xs: map(lambda x: 0 if x == "0" else 1, xs[1]),
        filter(lambda x: x[0] in experiment_indices,
               enumerate(split_binary_strings))):
    experiments.append(
        list(map(lambda x: x[1][x[0]], zip(on_off_values, parameters))))

with open('example_arrival_config.json.jinja2') as f:
    template = jinja2.Template(f.read())

    with open('example_arrival_config.json', 'w') as f2:
        f2.write(template.render(experiments=experiments))
