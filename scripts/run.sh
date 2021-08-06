#!/usr/bin/env bash

run_client() {
  python3 -m fltk single configs/cloud_experiment.yaml --rank=$(echo "-2+$(ip route get 1 | cut -f7 -d ' ' | cut -f4 -d '.'  | head -n 1)" | paste -sd+ | bc)
}

run_federator() {
  python3 -m fltk single configs/cloud_experiment.yaml --rank=0
}


while getopts "h?f?c?" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    f)
        echo "RUNNING FEDERATOR..."
        run_federator
        exit 0
        ;;
    c)  echo "RUNNING CLIENT..."
        run_federator
        exit 0
        ;;
    esac
done

shift $((OPTIND-1))
exit 0


