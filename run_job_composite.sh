#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=cannyedges_sketchguided_$1_$2
#SBATCH --partition=gpu-large
#SBATCH --output=cannyedges_sketchguided_$1_$2.out
#SBATCH --error=cannyedges_sketchguided_$1_$2.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pedrosergiot@lps.ufrj.br

cat /etc/hosts

singularity exec --bind /home/pedro.silva/ --nv \
/mnt/cern_data/micael.verissimo/images/base_conda.sif \
/home/pedro.silva/tese_ultrassom/sketch_guided_ultrasound_covid/composite_sketchguided_covid.sh \
$1 $2

EOT