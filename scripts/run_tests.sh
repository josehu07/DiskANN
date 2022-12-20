#!/bin/zsh

set -e

K=( 1 10 100 )
L=( 1 10 100 )

dataset="$1"
log_dir="$2"
ds_log="$log_dir/DS_time.log"
ds_ts_log="$log_dir/DS-TS-sync_time.log"
ds_ts_async_log="$log_dir/DS-TS-async_time.log"

rm -rf "$log_dir"
mkdir -p "$log_dir"

for k in "${K[@]}"; do
    for l in "${L[@]}"; do
        if [ "$l" -lt "$k" ]; then break; fi  # require: L >= K
        echo -e "\tK=$k L=$l"
        echo -n "K $k L $l " >> "$ds_log"
        {time /mnt/ssd/ann-remote/DiskANN/scripts/run.py query --dataset "$dataset" --k_depth $k --list_sizes $l &> "$log_dir/DS_K${k}-L${l}.log"} 2>> "$ds_log"
        echo -n "K $k L $l " >> "$ds_ts_log"
        {time /mnt/ssd/ann-remote/DiskANN/scripts/run.py query --dataset "$dataset" --k_depth $k --list_sizes $l --use_ts &> "$log_dir/DS-TS-sync_K${k}-L${l}.log"} 2>> "$ds_ts_log"
        echo -n "K $k L $l " >> "$ds_ts_async_log"
        {time /mnt/ssd/ann-remote/DiskANN/scripts/run.py query --dataset "$dataset" --k_depth $k --list_sizes $l --use_ts --ts_async &> "$log_dir/DS-TS-async_K${k}-L${l}.log"} 2>> "$ds_ts_async_log"
    done
done
echo -e "\n\nbase results:"
awk -F" " '{print $(2),$(4),$(NF-7),$(NF-5),$(NF-3),$(NF-1)}' "$ds_log" | tee "$log_dir/DS-res.txt"
echo -e "\n\nTS sync results:"
awk -F" " '{print $(2),$(4),$(NF-7),$(NF-5),$(NF-3),$(NF-1)}' "$ds_ts_log" | tee "$log_dir/DS-TS-sync-res.txt"
echo -e "\n\nTS async results:"
awk -F" " '{print $(2),$(4),$(NF-7),$(NF-5),$(NF-3),$(NF-1)}' "$ds_ts_async_log" | tee "$log_dir/DS-TS-async-res.txt"
