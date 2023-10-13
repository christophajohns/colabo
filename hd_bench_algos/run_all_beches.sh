#!/bin/bash
#SBATCH --time 10:00:00
#SBATCH --mem-per-cpu 4G
#SBATCH -n 2
#SBATCH --job-name error_run${1}_${2}
#SBATCH --time 4:00:00

#method_name =  sys.argv[1]
#function_name = sys.argv[2]
#seed = int(sys.argv[3])
#experiment_name = sys.argv[4]

EXPNAME=$1
# add rducb soon enough
for SEED in 1 2
do
    for MNAME in baxus mctsvs turbo cmaes
    do
        for FUNNAME in ${SYN}
        do
            sbatch run_bench.sh $MNAME $FUNNAME $SEED $EXPNAME
        done
    done
done

