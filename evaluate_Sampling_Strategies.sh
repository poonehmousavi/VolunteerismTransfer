#!/bin/sh
# Author : Pooneh Mousavi
# Script follows here:
INPUT_PATH="./data/PR_Labeled.json"
OUT_PATH="./result/"
GROUP_BY="eventid" # For grouping by specific event or event type <eventid,event_type>
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
SAMPLING_STRATEGIES=("up" "down" "up-with-same-eventtype" "up-with-same-eventCategory" "none")

#Evaluate Sampling Strategies without up-weighting ans save each sampling strategy result in separate file
#shellcheck disable=SC2068
for SAMPLING_STRATEGY in ${SAMPLING_STRATEGIES[@]}
do

  cp template.sbatch $SAMPLING_STRATEGY.sbatch

  for SEED in `cat ${SEEDS}`
  do
    for EVENT in ${EVENTS[@]}
    do
      echo python evaluate.py  \
        --inputpath=$INPUT_PATH   \
        --outputpath=$OUT_PATH$SAMPLING_STRATEGY$SUFFIX  \
        --heldout_event=$EVENT  \
        --seed_number=$SEED \
        --sampling_strategy=$SAMPLING_STRATEGY \
        --up_weighting=0 \
        --label_prop=0  \
        --groupby_col=$GROUP_BY >> $SAMPLING_STRATEGY.sbatch
        --random_sample=1
        --threshhold=0
    done
  done

  sbatch --partition datasci $SAMPLING_STRATEGY.sbatch &
done

#Evaluate Sampling Strategies with up-weighting ans save each sampling strategy result in separate file
SUFFIX2="-with-UP_WEIGHTING.json"
SAMPLING_STRATEGIES2=("up" "down" "none")
#shellcheck disable=SC2068
for SAMPLING_STRATEGY in ${SAMPLING_STRATEGIES2[@]}
do
  
  cp template.sbatch $SAMPLING_STRATEGY.weighted.sbatch

  for SEED in `cat ${SEEDS}`
  do
    for EVENT in ${EVENTS[@]}
    do
      echo python evaluate.py  \
        --inputpath=$INPUT_PATH   \
        --outputpath=$OUT_PATH$SAMPLING_STRATEGY$SUFFIX2  \
        --heldout_event=$EVENT  \
        --seed_number=$SEED \
        --sampling_strategy=$SAMPLING_STRATEGY \
        --up_weighting=1 \
        --label_prop=0  \
        --groupby_col=$GROUP_BY >> $SAMPLING_STRATEGY.sbatch
        --random_sample=1
        --threshhold=0
    done
  done

  sbatch --partition datasci $SAMPLING_STRATEGY.weighted.sbatch &
done


