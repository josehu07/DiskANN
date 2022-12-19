#!/bin/zsh

K=( 1 5 10 20 30 40 50 60 70 80 90 100 )

dataset=$1
log_dir=$2
ds_log=$log_dir/DS_time.log
ds_ts_log=$log_dir/DS-TS-sync_time.log
ds_ts_async_log=$log_dir/DS-TS-async_time.log

rm -rf $log_dir/*

for k in "${K[@]}"; do
    l_min=$(( $k > 10 ? $k : 10 ))
    for ((l=$l_min; l <=100; l+=10)); do
        echo -e "\tK=$k L=$l"
        echo -n "K $k L $l " >> $ds_log
        {time /mnt/ssd/ann-remote/DiskANN/scripts/run.py query --dataset $dataset --k_depth $k --list_sizes $l &> $log_dir/DS_K${k}-L${l}.log} 2>> $ds_log
        echo -n "K $k L $l " >> $ds_ts_log
        {time /mnt/ssd/ann-remote/DiskANN/scripts/run.py query --dataset $dataset --k_depth $k --list_sizes $l --use_ts &> $log_dir/DS-TS-sync_K${k}-L${l}.log} 2>> $ds_ts_log
        echo -n "K $k L $l " >> $ds_ts_async_log
        {time /mnt/ssd/ann-remote/DiskANN/scripts/run.py query --dataset $dataset --k_depth $k --list_sizes $l --use_ts --ts_async &> $log_dir/DS-TS-async_K${k}-L${l}.log} 2>> $ds_ts_async_log 
        #break
        #tail -n1 $log_dir/K${k}_${l}.log | p\awk -F" " '{print $k $l $(NF-7),$(NF-5),$(NF-3),$(NF-1)}' >> $log_dir/jct.txt
    done
    #break
done
echo -e "\n\nbase results:"
awk -F" " '{print $(2),$(4),$(NF-7),$(NF-5),$(NF-3),$(NF-1)}' $ds_log | tee $log_dir/DS-res.txt
echo -e "\n\nTS sync results:"
awk -F" " '{print $(2),$(4),$(NF-7),$(NF-5),$(NF-3),$(NF-1)}' $ds_ts_log | tee $log_dir/DS-TS-sync-res.txt
echo -e "\n\nTS async results:"
awk -F" " '{print $(2),$(4),$(NF-7),$(NF-5),$(NF-3),$(NF-1)}' $ds_ts_async_log | tee $log_dir/DS-TS-async-res.txt
