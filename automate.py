import subprocess
import itertools
import json

CLUSTER_NAMES = ['fltk-cluster'] #all the clusters to vary factor C - performance
PROJECT_NAME = 'qpecs-main'
N_REPS = 5
ML_MODELS = ['ResNet18', 'ResNet152']
PARALLELISM = [1,4]
DATASET = 'CIFAR10'
CONFIG_FILE = 'configs/tasks/example_arrival_config.json'

combinations = list(itertools.product(ML_MODELS, CLUSTER_NAMES, PARALLELISM, range(N_REPS)))

for model, cluster, n_parallel, i in combinations:

    # fetch the credentials of the cluster using googlke cloud SDK
    # the command will fet the credentials of the cluster and put in it ~/.kube
    cmd = 'gcloud container clusters get-credentials {} \
            --zone us-central1-c --project {}'.format(cluster, PROJECT_NAME)
    subprocess.run(cmd, shell=True)


    # editing the json config file to vary factor B (model) and C (CPU)
    with open(CONFIG_FILE, 'r') as json_file:
        conf = json.load(json_file)
        conf[0]['jobClassParameters'][0]['networkConfiguration']['network'] = model
        conf[0]['jobClassParameters'][0]['networkConfiguration']['dataset'] = DATASET
        conf[0]['jobClassParameters'][0]['systemParameters']['dataParallelism'] = str(n_parallel)

    with open(CONFIG_FILE, 'w') as json_file:
        json.dump(conf, json_file, indent=4)


    # building the docker container
    cmd = 'DOCKER_BUILDKIT=1 docker build . --tag gcr.io/{}/fltk'.format(PROJECT_NAME)
    subprocess.run(cmd, shell=True)


    # pushing the docker container
    cmd = 'docker push gcr.io/{}/fltk'.format(PROJECT_NAME)
    subprocess.run(cmd, shell=True)


    # cd into charts and install the extractor
    cmd = 'helm upgrade --install extractor ./extractor -f fltk-values.yaml --namespace test'
    subprocess.run(cmd, shell=True, cwd='charts/')


    # finally launch an experiment
    cmd = 'helm upgrade --install orchestrator ./orchestrator --namespace test -f fltk-values.yaml'
    subprocess.run(cmd, shell=True, cwd='charts/')


    input("Running. Press any key to continue to next iteration...")
