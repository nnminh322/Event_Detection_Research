# currently kmean get distribution
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -e
log_p=KDR/maven
mkdir -p $log_p
#cp -r  $BASH_SOURCE $log_p/run.sh

for i in  0 1 2 3 4
do  
    if [ ! -d  "${log_p}/perm_${i}" ]; 
    then
        mkdir -p ${log_p}/perm_${i}
    fi

    touch ${log_p}/perm_${i}/exp.log
    cp -r models/nets.py ${log_p}/perm_${i}/
    cp -r utils/datastream.py ${log_p}/perm_${i}/
    cp -r utils/worker.py ${log_p}/perm_${i}/

    python run_train.py --log-dir "${log_p}/perm_${i}"  --perm-id ${i} --dropout "normal" --p 0.2 --mul_distill --kt --kt2 --train-epoch 15 --patience 5  --generate  --batch-size 128 --mode herding --clusters 4 --mul_task --num_sam_loss 2 > ${log_p}/perm_${i}/exp.log


done
