#!/bin/sh
# Author : Pooneh Mousavi
INPUT_PATH="/data/"
FILENAMES=("PR_Random+Labeled_Data.json" "PR_NGO+Labeled_Data" "PR_NGO+Labeled_Data")
OUT_PATH="/result/"
GROUP_BY="eventid"
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
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
SAMPLING_STRATEGIES="down"
UP_WEIGHTING=0

#Evaluate Weak Supervison using Label Spreading
#shellcheck disable=SC2068
for FILENAME in ${FILENAMES[@]}
do
  for SEED in ${SEEDS[@]}
  do
    for EVENT in ${EVENTS[@]}
    do
      python evaluate.py  --inputpath=$INPUT_PATH$FILENAME$SUFFIX   --outputpath=$OUT_PATH$F$FILENAME$SUFFIX  --heldout_event=$EVENT  --seed_number=$SEED --sampling_strategy=$SAMPLING_STRATEGY --up_weighting=$UP_UP_WEIGHTING --label_spreading=1  --groupby_col=$GROUP_BY

    done
  done
done

#Evaluate Weak Supervison without  Label Spreading
FILENAMES2=("PR_NGO+Labeled_Data")
SUFFIX2="_without_label_spreading"
#shellcheck disable=SC2068
for FILENAME in ${FILENAMES2[@]}
do
  for SEED in ${SEEDS[@]}
  do
    for EVENT in ${EVENTS[@]}
    do
      python evaluate.py  --inputpath=$INPUT_PATH$FILENAME$SUFFIX   --outputpath=$OUT_PATH$F$FILENAME$SUFFIX2$SUFFIX  --heldout_event=$EVENT  --seed_number=$SEED --sampling_strategy=$SAMPLING_STRATEGY --up_weighting=$UP_UP_WEIGHTING --label_spreading=1  --groupby_col=$GROUP_BY

    done
  done
done
