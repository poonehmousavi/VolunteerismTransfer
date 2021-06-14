# VolunteerismTransfer
Please Donate for the Affected'': Supporting Emergency Managers in Finding Volunteers and Donations in Twitter Across Disasters

Despite the outpouring of social support posted to social media channels in the aftermath of disaster, finding and managing content that can translate into community relief, donations, volunteering, or other recovery support is difficult.
This paper outlines methods for providing computational support in this emergency support function by evaluating the consistency of language in volunteer- and donation-related social media content across 31 disasters and developing semi-automated models for classifying volunteer- and donation-related social media content in new disaster events.
Results show volunteer- and donation-related social media content is sufficiently similar across disaster events and disaster types to warrant transferring models across disasters, and we describe a simple resampling technique for tuning these models for specific types of disasters.
We then introduce and evaluate a weak-supervision approach to integrate domain knowledge from emergency response officers with machine learning models to improve classification accuracy and accelerate this emergency support in new events.

The project contains three main files:

  1)CollectData: Use to collect Data from 4 different sources: 
  
                  -TrecIS 2018-2019-2020A (42922 tweets)
                  -CrisiNLP (47155 tweets)
                  -NGO(5 Top Accounts) (16080 tweets)
                  -Random Trec IS Data (59734 tweets)
    
   Finally, we generate 4 different set of Data from combination of above sources. The format of the files is all the same contains "postID","eventid","event_type","text","volunteerLabels","src" :
                 
                 -Labeled Data: TrecIS 2018-2019-2020A+CrisiNLP
                 -NGO+Labeled Data
                 -Random+Labeled Data
                 -NGO+Random+Labeled Data
                  
  2)Analyze-data: 
  
    This file contains all functions necessary for analyzing data. First, it applies common processing techniques to a tweet's text including removing stopwords, lemmatization,     lowercase, removing special characters, and tokenize it. We save the processed data in a separate column as "processed_text" to use in the future for either simple tfidf or     word embedding. Second, it contains an LDA topic modeling.  Third, it contains sentence-bert embedding for generating a similarity matrix for assigning weights to different     samples during training. Finally, It contains a call to classification pipeline that trains and evaluates different sampling strategies.
  
                
  
  3)evaluate: This file contains code for training SVM model with different balancing strategies.First you should choose any data from previous step.Then Three     things need to be initiated. 
  
                -groupby_col. Two options chould be selected for this value: "eventid" and "event_type" . It Specifies how to split training and test data.
                -sampling_strategy . It specifies sampling strategy. It could be:
                
         
                        -"none": imbalanced data
                        -"up" :upsampling
                        -"down": downsampling
                        -'up-with-same-eventtype':  when up-sampling, resample primarily from events of the same type
                        -'up-with-same-eventCategory': same event-type with the highest weight, data of the same “kind” of event (manmade vs. natural) weighted~6, and annotated data of other “kinds” of events weighted ~3)
                        -up-with-similarity-eventCategory: when up-sampling, assign weights according to the similarity matrix

                -up_weighting:could be true or false. It could be true for sampling_strategy : "none","up" and "down".If true, it is up-weighting samples from the same event       data.
