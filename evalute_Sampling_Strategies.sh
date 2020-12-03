#!/bin/sh
# Author : Pooneh Mousavi
# Script follows here:
INPUT_PATH="/data/PR_Labeled.json"
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
SAMPLING_STRATEGIES=("none" "up" "down" "up-with-same-eventtype" "up-with-same-eventCategory")

#Evaluate Sampling Strategies without up-weighting ans save each sampling strategy result in separate file
#shellcheck disable=SC2068
for SAMPLING_STRATEGY in ${SAMPLING_STRATEGIES[@]}
do
  for SEED in ${SEEDS[@]}
  do
    for EVENT in ${EVENTS[@]}
    do
      python evaluate.py  --inputpath=$INPUT_PATH   --outputpath=$OUT_PATH$SAMPLING_STRATEGY$SUFFIX  --heldout_event=$EVENT  --seed_number=$SEED --sampling_strategy=$SAMPLING_STRATEGY --up_weighting=0 --label_spreading=0  --groupby_col=$GROUP_BY
    done
  done
done

#Evaluate Sampling Strategies with up-weighting ans save each sampling strategy result in separate file
SUFFIX2="-with-UP_WEIGHTING.json"
SAMPLING_STRATEGIES2=("none" "up" "down")
#shellcheck disable=SC2068
for SAMPLING_STRATEGY in ${SAMPLING_STRATEGIES2[@]}
do
  for SEED in ${SEEDS[@]}
  do
    for EVENT in ${EVENTS[@]}
    do
      python evaluate.py  --inputpath=$INPUT_PATH   --outputpath=$OUT_PATH$SAMPLING_STRATEGY$SUFFIX2  --heldout_event=$EVENT  --seed_number=$SEED --sampling_strategy=$SAMPLING_STRATEGY --up_weighting=1 --label_spreading=0  --groupby_col=$GROUP_BY
    done
  done
done
