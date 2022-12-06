#!/bin/bash

#args: fvec_file (#max_npts)
to_bin() {
    fn=$1
    max_npts=$2
    if [ "$max_npts" != "" ]; then
        echo "debug tiny: trimming to $max_npts points"
        ~/DiskANN/build/tests/utils/fvecs_to_bin "$fn"_base.fvecs  /mnt/ssd/data/sift-tiny/sifttiny_learn.fbin $max_npts
    else
        ~/DiskANN/build/tests/utils/fvecs_to_bin "$fn"_base.fvecs "$fn".fbin
    fi
}

# args: binary_file
run_build () {
    fn=$1
    #use_debug=$2
    ~/DiskANN/build-debug/tests/build_disk_index --data_type float --dist_fn l2 --data_path "$fn".fbin --index_path_prefix "$fn"_R32_L50_A1.2 -R 32 -L50 -B 0.003 -M 1
}

#args: index_file use_debug use_ts
run_search () {
    fn=$1
    use_debug=$2
    use_ts=$3

    mkdir $fn
    ./tests/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix "$fn"_learn_R32_L50_A1.2 --query_file fn_query.fbin  --gt_file "$fn"_query_learn_gt100 -K 10 -L 10 20 30 40 50 100 --result_path "$fn"/res --num_nodes_to_cache 10000
}

if [ "$1" = "to_bin" ]; then
    echo "to binary"
    to_bin $2 $3
elif [ "$1" = "build" ]; then
    echo "build index"
    run_build $2
    #run_build $2 $3
elif [ "$1" = "search_original" ]; then
    echo "test search"
    run_search $2 $3 $4
fi
