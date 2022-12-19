import argparse
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kl_pairs = [(1,10), (10,10), (10,100)]

def extract(fname):
    # trimming
    def toNum(line):
        for i in range(0,2):
            line[i] = int(line[i])
        for i in range(2,5):
            line[i] = float(line[i][:-1])
        line[5] = float(line[5])
        return line
    with open(fname, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines] #[K	L user-time kernel-time CPU-usage JCT]
        lines = list(map(toNum, lines))
    return lines

def draw_and_save(extracted, labels, output_dir):
    # format data
    dps = []
    for i in range(len(extracted)):
        dps.append( [0] * len(kl_pairs) )

    for i in range(len(kl_pairs)):
        pair = kl_pairs[i]
        # extract correspodning jcts
        for j in range(len(extracted)):
            lines = list(filter(lambda line: line[0] == pair[0] and line[1] == pair[1], extracted[j])) #should be just 1 line
            dps[j][i] = lines[0][5]
    # swap to match label sequence (going unportable, too much effort)
    dps[0],dps[1],dps[2] = dps[1],dps[2],dps[0]
    
    # plotting
    x = np.arange(len(kl_pairs))
    width = 0.6 / len(extracted) #bar width

    fig, ax = plt.subplots()

    for i in range(len(extracted)):
        #calculate bar location relative to group centroid
        shift =  (i * 2 + 1 - len(extracted)) / 2
        pos = x + width * shift
        rect = ax.bar(pos, dps[i], width, label=labels[i])
        ax.bar_label(rect, fmt="%.3g", padding=3)

    ax.set_ylabel('JCT/s')
    ax.set_title('Job Completion Times by K and L')
    ax.set_xticks(x, kl_pairs)
    ax.legend()

    fig.tight_layout()

    save_fn = os.path.join(output_dir, str(datetime.datetime.now())+'.png')
    plt.savefig(save_fn)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver of experiment plots")
    parser.add_argument('--input_dir', help="input directory, where the -res.txt files reside", required=True)
    parser.add_argument('--output_dir', help="output directory for the plots", required=True)
    
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    input_files = os.listdir(input_dir+'/')
    input_files = list(filter(lambda fn: str(fn).endswith('-res.txt'), input_files))
    print(f'input files{input_files}')
    input_file_full_paths = [ os.path.join(input_dir, fn) for fn in input_files ]
    
    extracted = [ extract(fn) for fn in input_file_full_paths ]
    
    labels = ['base', 'TensorStore', 'TensorStore+AsyncIO']
    draw_and_save(extracted, labels, args.output_dir)

