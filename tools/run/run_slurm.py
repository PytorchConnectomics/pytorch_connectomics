# Automatically generate slurm scripts for running inference on RC cluster 
import os, sys

def get_pref(mem=10000, do_gpu=False):
    pref = '#!/bin/bash\n'
    pref+= '#SBATCH -N 1 # number of nodes\n'
    pref+= '#SBATCH -p cox\n'
    pref+= '#SBATCH -n 2 # number of cores\n'
    pref+= '#SBATCH --mem '+str(mem)+' # memory pool for all cores\n'
    if do_gpu:
        pref+= '#SBATCH --gres=gpu:2 # memory pool for all cores\n'
    pref+= '#SBATCH -t 3-00:00:00 # time (D-HH:MM)\n'
    pref+= '#SBATCH -o logs/deploy_%j.log\n\n'
    pref+= 'module load cuda\n'
    pref+= 'source activate py3_torch\n\n'
    return pref

def gen_slurm(Do = '/path/to/slurm/'):
    cmd=[]
    mem=50000
    do_gpu= True

    fn='deploy' # output file name
    suf = '\n'
    num = 32
    cn = 'deploy.py'
    cmd+=['python -u /path/to/script/'+cn+' %d '+str(num)+suf]

    pref=get_pref(mem, do_gpu)

    if not os.path.exists(Do):
        os.makedirs(Do)

    for i in range(num):
        a=open(Do + fn+'_%02d.sh'%(i),'w')
        a.write(pref)
        for cc in cmd:
            if '%' in cc:
                a.write(cc%i)
            else:
                a.write(cc)
        a.close()
