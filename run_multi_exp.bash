#!/bin/bash

## declare an array variable
declare -a arr=("configs/experiment_vanilla.yaml"
                # "configs/experiment_deadline.yaml"
                # "configs/experiment_swyh.yaml"
                # "configs/experiment_freeze.yaml"
                # "configs/experiment_offload.yaml"
                )
EVENT_FILE="exp_events.txt"
# Check if all files are present
for i in "${arr[@]}"
do
#   echo "$i"
   if [ ! -f $i ]; then
      echo "File not found! Cannot find: $i"
#      exit
  fi
   # or do whatever with individual element of the array
done

read -p "Do you wish to continue? (y/n)?" choice
case "$choice" in
  y|Y ) ;;
  n|N ) exit;;
  * ) exit;;
esac

echo "" > $EVENT_FILE

# Start running experiments
## now loop through the above array
for i in "${arr[@]}"
do
  export EXP_CONFIG_FILE="$i"
  echo "[$(date +"%T")] Starting $EXP_CONFIG_FILE"
  echo "[$(date +"%T")] Starting $EXP_CONFIG_FILE" >> $EVENT_FILE
  start_time=$(date +%s)
  docker-compose up --build 2>&1 | tee dc_log.txt
  end_time=$(date +%s)
  # elapsed time with second resolution
  elapsed=$(( end_time - start_time ))
  echo "[$(date +"%T")] Finished with $EXP_CONFIG_FILE in $elapsed seconds" >> $EVENT_FILE
#   docker-compose up
   # or do whatever with individual element of the array
done
echo "[$(date +"%T")] Finished all experiments"
echo "[$(date +"%T")] Finished all experiments" >> $EVENT_FILE
