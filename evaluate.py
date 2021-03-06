import argparse
import json

import gensim.downloader as api
import pandas as pd

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score, precision_score, recall_score, \
    precision_recall_curve

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from tqdm import tqdm
from sklearn.metrics import plot_confusion_matrix
import os
from gensim.models import KeyedVectors
from gensim.downloader import base_dir

import gensim.downloader as api
class Label_Prop_Approach(enum.Enum):
   none = 0
   all_volunteer = 1
   label_spreading = 2
   semi_supervised = 3

pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb', SGDClassifier(loss='hinge')),
])



def read_file(file_path):
    data = pd.read_json(file_path, orient='records', lines=True)
    return data


def get_sample_category(event_type):
    if 'earthquake' in event_type or "hurricane/typhoon/cyclone/tornado" in event_type or "flood" in event_type or "wildfire/bushfire" in event_type or "outbreak" in event_type:
        return 'natural'
    elif "bombing" in event_type or "shooting" in event_type or "explosion" in event_type or "collapse" in event_type:
        return "manmade"
    elif "general" in event_type:
        return "general"
    elif "random" in event_type:
        return 'random'



def get_sample_weight(event_type, target):

    if event_type == target:
        return 10
    elif get_sample_category(event_type) == get_sample_category(target):
        return 6
    elif get_sample_category(event_type)=='random':
        return 1
    elif get_sample_category(event_type) == 'general':
        return 3
    else:
        return 5


def get_climate_related(event_type):
    if 'hurricane/typhoon/cyclone/tornado' in event_type:
        return 1
    elif 'flood' in event_type:
        return 1
    elif 'wildfire/bushfire' in event_type:
        return 1
    else:
        return 0



def get_sample_climate_weight(event_type, target):

    if event_type == target:
        return 10
    elif get_climate_related(event_type) == 1 :
        return 6
    elif get_sample_category(event_type)=='natural':
        return 3
    else:
        return 1

def get_sample_climate_weight_similariy(event_type, target):
    if event_type == target:
        return 10
    if event_type in similarity_matrix.index:
        return similarity_matrix.loc[target][event_type]*10
    else:
        return 1

def get_sample_data(data, seed_number, label_prop_approach,threshhold,random_sample=True):

    if random_sample == True:
        labeled_data = data.query("src == 'trec' or src == 'crisis_nlp'")
        other = data.query("src != 'trec' and src != 'crisis_nlp'")
        N = len(labeled_data)
        sample_data = labeled_data.sample(n=N, random_state=seed_number, replace=True)
        sample_data = sample_data.append(other)
    else:
        sample_data=data
    if label_prop_approach == Label_Prop_Approach.label_spreading.value :
        sample_data['ft_features'] = [vectorize(str(s)) for s in sample_data["processed_text"]]
        sample_data['l'] = sample_data.apply(lambda row: row['ft_features'].size, axis=1)
        sample_data = sample_data.query("l >1")

        sample_data.drop(columns=['l'], inplace=True)
        X_data = sample_data['ft_features']
        y_labels = sample_data['label'].copy()
        y_labels[sample_data['src'].isin(['top-accounts', 'random'])] = -1
        label_prop_model = LabelSpreading(kernel='knn', n_jobs=-1)
        y_learned = label_prop_model.fit(X_data.tolist(), y_labels).transduction_
        sample_data["label_spread"] = y_learned
        sample_data = sample_data.query(
            "src == 'trec' or src=='crisis_nlp' or ((src=='top-accounts' or src=='random') and label_spread==1 )")
    if label_prop_approach == Label_Prop_Approach.semi_supervised.value:
        sample_data['ft_features'] = [vectorize(str(s)) for s in data["processed_text"]]
        sample_data['l'] = data.apply(lambda row: row['ft_features'].size, axis=1)
        sample_data = data.query("l >1")
        sample_data.drop(columns=['l'], inplace=True)

        training = sample_data.query("src == 'trec' or src == 'crisis_nlp'")
        test = sample_data.query("src != 'trec' and src != 'crisis_nlp'")
        vol = training.loc[training['label'] == 1]
        non_vol = training.loc[training['label'] == 0]
        training_up = non_vol.append(vol.sample(n=len(non_vol), replace=True), ignore_index=True)

        X_train = training_up['ft_features']
        X_test = test['ft_features']
        y_train = training_up['label']
        pipeline_lr = Pipeline([
            ('nb', SGDClassifier(loss='log')),
            ])
        model = pipeline_lr.fit(X_train.tolist(), y_train)
        test['label_prob'] = model.predict_proba(X_test.tolist())[:, 1]
        test = test.loc[test['label_prob'] >= threshhold]

        test.drop(columns=['label_prob'], inplace=True)
        test['label'] = 1
        sample_data = training.append(test)

    return sample_data

def evaluate_model(data, heldout_event,groupby_col, sampling_strategy, up_weighting, seed_number, label_prop_approach,random_sample=True,threshhold=0.5):
    # Decide with field to use for training as a gold-label
    if label_prop_approach == Label_Prop_Approach.label_spreading.value:
        lab = 'label_spread'
    else:
        lab = 'label'

    # Split Test and trainig data based on heldout_event/heldout_eventType  and get the resample training data
    main_training = data[data[groupby_col] != heldout_event]
    test = data[data[groupby_col] == heldout_event]
    training = get_sample_data(main_training, seed_number, label_prop_approach,threshhold,random_sample=random_sample)


    # Randomely give weights to each sample,then set higher weights to event of same-type
    if up_weighting == True:
        training['sample_weight'] = 100 * np.abs(np.random.randn(training.shape[0]))
        training['sample_weight'] = np.where(training['event_type'] == test.iloc[0].event_type,
                                             training['sample_weight'] * 10,
                                             training['sample_weight'])
    # Split training data to volunteer and non-volunterrs for further use in applying sampling strategies
    vol = training.loc[training[lab] == 1]
    non_vol = training.loc[training[lab] == 0]

    # Upsampling Strategy
    if sampling_strategy == 'up':
        training = non_vol.append(vol.sample(n=len(non_vol), replace=True), ignore_index=True)

    # Downsampling Strategy
    elif sampling_strategy == 'down':
        training = vol.append(non_vol.sample(n=len(vol), replace=False), ignore_index=True)

    # when up-sampling, resample primarily from events of the same type
    elif sampling_strategy == "up-with-same-eventtype":
        training['sample_weight'] = 1
        training['sample_weight'] = np.where(training['event_type'] == test.iloc[0].event_type,
                                             training['sample_weight'] * 10,
                                             training['sample_weight'])
        vol = training.loc[training[lab] == 1]
        non_vol = training.loc[training[lab] == 0]
        training = non_vol.append(vol.sample(n=len(non_vol), weights='sample_weight', replace=True), ignore_index=True)
    # when up-sampling, resample primarily from events of the same “kind” of event (manmade vs. natural)
    elif sampling_strategy == "up-with-same-eventCategory":
        training['sample_weight'] = [get_sample_climate_weight(x, test.iloc[0].event_type) for x in training['event_type']]
        vol = training.loc[training[lab] == 1]
        non_vol = training.loc[training[lab] == 0]
        training = non_vol.append(vol.sample(n=len(non_vol), weights='sample_weight', replace=True), ignore_index=True)
    elif sampling_strategy == "up-with-similarity-eventCategory":
        training['sample_weight'] = [get_sample_climate_weight_similariy(x, test.iloc[0].event_type,) for x in training['event_type']]
        vol = training.loc[training[lab] == 1]
        non_vol = training.loc[training[lab] == 0]
        training = non_vol.append(vol.sample(n=len(non_vol), weights='sample_weight', replace=True), ignore_index=True)

    X_train = [str(x) for x in training['processed_text']]
    X_test = [str(x) for x in test['processed_text']]
    y_train = training[lab]
    y_test = test['label']
    if up_weighting == True:
        model = pipeline_sgd.fit(X_train, y_train, nb__sample_weight=training['sample_weight'])
    else:
        model = pipeline_sgd.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    recall = recall_score(y_test, y_predict, zero_division=0)
    precision = precision_score(y_test, y_predict, zero_division=0)
    from sklearn.metrics import f1_score
    f1_score = f1_score(y_test, y_predict, zero_division=0)
    return {groupby_col: test.iloc[0].eventid + '-' + str(seed_number), 'src': test.iloc[0].src, 'precision': precision,
            'recall': recall, 'f1_score': f1_score}


def get_different_sampling_strategy(data,seed,heldout_event):
    performance_result={}
    sampling_strategies=['none','up','down','up-with-same-eventtype','up-with-same-eventCategory']
    for sampling_strategy in tqdm(sampling_strategies):
        if sampling_strategy== 'none' or sampling_strategy=='up' or sampling_strategy=='down':
            performance_result[sampling_strategy+'-without-upweight']=evaluate_model(data,heldout_event,sampling_strategy,up_weighting=False,seed_number=seed)
            performance_result[sampling_strategy+'-with-upweight']=evaluate_model(data,heldout_event,sampling_strategy,up_weighting=True,seed_number=seed)
        else:
            performance_result[sampling_strategy]=evaluate_model(data,heldout_event,sampling_strategy,up_weighting=False,seed_number=seed)
    return performance_result

def vectorize(sentence):

    tokenized = [t for t in analyzer(sentence)]

    wv_vecs = []
    for t in tokenized:

        try:
            v = wvs[t]
            norm = np.linalg.norm(v)
            normed_v = (v / norm)
            wv_vecs.append(normed_v)
        except:
            continue

    m = np.array(wv_vecs)
    normed_m = np.mean(m, axis=0)

    normed_m = np.nan_to_num(normed_m)


    return normed_m



def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath',  type=str,required=True,
                        help='input file path where all model are saved ')
    parser.add_argument("--outputpath", type=str, required=True,
                        help="outputh file path to save the result")
    parser.add_argument("--seed_number", type=int, required=True,
                        help="the seed number for random number generator")
    parser.add_argument("--heldout_event", type=str, required=True,
                        help="name of the event which is held-out for test ")
    parser.add_argument("--sampling_strategy", type=str, required=True,
                        help="Specify the balancing strategies ")
    parser.add_argument("--up_weighting", type=int,  default=0,
                        help="Specify wheter to add more weight to sample fo the same event_type as heldout_event ")
    parser.add_argument("--label_prop", type=int, default=0,
                        help="Specify to apply label spreading or not")
    parser.add_argument("--groupby_col", type=str, default='eventid',
                        help="Could be either eventid or eventtype for generating held-out event accordingly ")
    parser.add_argument("--random_sample", type=int, default=1,
                        help="Specify to use random sample of data using seed number or all the training ")
    parser.add_argument("--threshhold", type=float, default=1,
                        help="Specify to use random sample of data using seed number or all the training ")


    return parser



def main():
    global analyzer
    global wvs
    global vectorizer
    global similarity_matrix
    parser = read_parameters()
    args = parser.parse_args()
    inputpath=args.inputpath
    outputpath=args.outputpath
    heldout_event=args.heldout_event
    seed_number = args.seed_number
    sampling_strategy = args.sampling_strategy
    up_weighting = False if args.up_weighting==0 else True
    label_prop=args.label_prop
    # random_sampling = False if args.up_weighting == 0 else True
    random_sampling=True
    threshhold = args.threshhold
    groupby_col=args.groupby_col

    if sampling_strategy == "up-with-similarity-eventCategory":
        similarity_matrix=pd.read_csv("./event_ranks.csv")
        similarity_matrix=similarity_matrix.rename(columns={"reference-event":'event'})
        similarity_matrix=similarity_matrix.set_index("event")

    if label_prop==Label_Prop_Approach.label_spreading.value or label_prop==Label_Prop_Approach.semi_supervised.value:
        print("Start Loading Word Embedding ")
        # print(api.load('fasttext-wiki-news-subwords-300', return_path=True))
        path = os.path.join(base_dir, 'fasttext-wiki-news-subwords-300', 'fasttext-wiki-news-subwords-300.gz')
        model_gensim = KeyedVectors.load_word2vec_format(path)
        wvs = model_gensim.wv
        vectorizer = TfidfVectorizer(
            use_idf=True,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=10000,
            min_df=4,
            max_df=0.501
        )
        analyzer = vectorizer.build_analyzer()
        print("End Loading Word Embedding ")

    print(f"Start evanulating model for sampling strategy: {sampling_strategy} with up-weighting={up_weighting}  and  label spreading={Label_Prop_Approach(label_prop).name} on {heldout_event} with seed number {seed_number}")
    data=read_file(inputpath)
    result=evaluate_model(data,heldout_event=heldout_event,groupby_col=groupby_col,sampling_strategy=sampling_strategy,up_weighting=up_weighting,seed_number=seed_number,label_prop_approach=label_prop,threshhold=threshhold ,random_sample=random_sampling)
    print(f"finish evanulating model for sampling strategy: {sampling_strategy} with up-weighting={up_weighting}  and  label spreading={Label_Prop_Approach(label_prop).name} on {heldout_event} with seed number {seed_number}")

    result_file = open(outputpath, "a+", encoding='utf-8')
    result_file.write(json.dumps(result) + '\n')
    result_file.close()




if __name__ == "__main__":
    main()
