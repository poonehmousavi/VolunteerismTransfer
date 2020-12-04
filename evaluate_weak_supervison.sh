#!/bin/sh
# Author : Pooneh Mousavi
INPUT_PATH="./data/"
FILENAMES=("PR_Random+Labeled_Data" "PR_NGO+Labeled_Data" "PR_NGO+Random+Labeled_Data")
OUT_PATH="./result_ls/"
GROUP_BY="eventid"
SEEDS="seeds.txt" # Cody will change that
EVENTS=('albertaFloods2013'
 'albertaWildfires2019'
 'australiaBushfire2013'
 'cycloneKenneth2019'
 'fireYMM2016'
 'hurricaneFlorence2018'
 'manilaFloods2013'
 'nepalEarthquake2015'
 'parisAttacks2015'
 'philipinnesFloods2012'
 'southAfricaFloods2019'
 'typhoonHagupit2014'
 'typhoonYolanda2013')
SUFFIX=".json"
SAMPLING_STRATEGY="up"
UP_WEIGHTING=0

#Evaluate Weak Supervison using Label Spreading
for FILENAME in ${FILENAMES[@]}
do

  cp template.sbatch $FILENAME.sbatch

  for SEED in `cat ${SEEDS}`
  do
    for EVENT in ${EVENTS[@]}
    do
      echo python evaluate.py  \
        --inputpath=$INPUT_PATH$FILENAME$SUFFIX   \
        --outputpath=$OUT_PATH$F$FILENAME$SUFFIX  \
        --heldout_event=$EVENT  \
        --seed_number=$SEED \
        --sampling_strategy=$SAMPLING_STRATEGY \
        --up_weighting=$UP_WEIGHTING \
        --label_spreading=1  \
        --groupby_col=$GROUP_BY >> $FILENAME.sbatch

    done
  done

  sbatch --partition datasci $FILENAME.sbatch &
done

#Evaluate Weak Supervison without  Label Spreading
FILENAMES2=("PR_NGO+Labeled_Data")
SUFFIX2="_without_label_spreading"
for FILENAME in ${FILENAMES2[@]}
do

  cp template.sbatch $FILENAME.nols.sbatch

  for SEED in `cat ${SEEDS}`
  do
    for EVENT in ${EVENTS[@]}
    do
      echo python evaluate.py  \
        --inputpath=$INPUT_PATH$FILENAME$SUFFIX   \
        --outputpath=$OUT_PATH$F$FILENAME$SUFFIX2$SUFFIX  \
        --heldout_event=$EVENT  \
        --seed_number=$SEED \
        --sampling_strategy=$SAMPLING_STRATEGY \
        --up_weighting=$UP_WEIGHTING \
        --label_spreading=0  \
        --groupby_col=$GROUP_BY >> $FILENAME.nols.sbatch

    done
  done

  sbatch --partition datasci $FILENAME.nols.sbatch &
done
