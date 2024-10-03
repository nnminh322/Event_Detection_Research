# currently kmean get distribution
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -e
log_p=KDR/ace
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

    python run_train.py --feature-root "data/ace_features" --llm2vec --stream-file "data/ACE/streams.json" --datasetname ACE --log-dir "${log_p}/perm_${i}"  --perm-id ${i} --dropout "normal" --p 0.2 --mul_distill --kt --kt2 --train-epoch 30 --patience 6 --init-slots 13  --generate --generate_ratio 20 --naloss_ratio 40  --max-slot 34 --batch-size 256 --mode herding --clusters 4 --mul_task --num_sam_loss 2 > ${log_p}/perm_${i}/exp.log

done
