import os


def gen_all_scripts(sname='', script='', params=[]):
    ibex_row1 = "#!/bin/bash\n#SBATCH -N 1\n#SBATCH --partition=batch"
    ibex_row2 = f'#SBATCH -J {sname}\n'
    ibex_row3 = f'#SBATCH -o {sname}.%J.out\n'
    ibex_row4 = f'#SBATCH -e {sname}.%J.err\n'
    ibex_row5 = f'#SBATCH --mail-user=rui.li@kaust.edu.sa\n'
    ibex_row6 = '#SBATCH --mail-type=ALL\n'
    ibex_row7 = '#SBATCH --time=23:30:00\n'
    ibex_row8 = '#SBATCH --mem=50G\n'
    ibex_row9 = '#SBATCH --gpus=1\n'
    ss = ibex_row1
    ss += ibex_row2
    ss += ibex_row3
    ss += ibex_row4
    ss += ibex_row5
    ss += ibex_row6
    ss += ibex_row7
    ss += ibex_row8
    ss += ibex_row9
    ss += script
    return ss


if __name__ == '__main__':
    ibex_system = '#!/bin/bash'
    print(ibex_system)
    rootpath = '/home/lir0b/Code/NeuralRep/NIR-3Drec/dependencies/IRON/data_flashlight'
    datalist = os.listdir(rootpath)

    for folder in datalist:
        script = f'. ./train_synthetic.sh {folder} 0'
        content = gen_all_scripts(sname=folder, script=script)
        print(content)
        if True:
            with open(f'train_syn_{folder}.sh', 'w+') as f:
                f.write(content)

    ss = ''
    for folder in datalist:
        ss += f'sbatch train_syn_{folder}.sh\n'
    with open(f'train_syn_all.sh', 'w+') as f:
        f.write(ss)
