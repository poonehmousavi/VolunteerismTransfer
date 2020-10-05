# VolunteerismTransfer
Please Donate for the Affected'': Supporting Emergency Managers in Finding Volunteers and Donations in Twitter Across Disasters

Despite the outpouring of social support posted to social media channels in the aftermath of disaster, finding and managing content that can translate into community relief, donations, volunteering, or other recovery support is difficult.
This paper outlines methods for providing computational support in this emergency support function by evaluating the consistency of language in volunteer- and donation-related social media content across 31 disasters and developing semi-automated models for classifying volunteer- and donation-related social media content in new disaster events.
Results show volunteer- and donation-related social media content is sufficiently similar across disaster events and disaster types to warrant transferring models across disasters, and we describe a simple resampling technique for tuning these models for specific types of disasters.
We then introduce and evaluate a weak-supervision approach to integrate domain knowledge from emergency response officers with machine learning models to improve classification accuracy and accelerate this emergency support in new events.

The project contains three files:

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
                  
  2)Preprocess: This file contains three parts. First, we apply usual preprocessing techniques on tweet's text including removing stopwords,lemmatization,           lowercase, removing speciall characters and tokenize it. We  save the processed data in a separate column as "processed_text" to use in future 
    for either simple tfidf or word embedding. Files contain "PR" prefix contain the "processed_text" column.
    Second part is for applying pretrained word embedding. You could load your preferred word-embedding and load the file with "PR" prefix and apply the word         embedding. Since the process is time consuming, we save a result in separate column called "ft_features" and we save the file using "FT" prefix.
    Last part is about label spreading, we load the data from files with "FT" prefix and apply label spreading on them and save the result in files with "LS"         prefix. At the end of this phase, we need to have 5 final files for next step:
    
                -FT_Labeled_Data.json
                -FT_NGO_Labeled_Data.json
                -LS_NGO_Labeled_Data.json
                -LS_Random_Labeled_Data.json
                -LS_Random_NGO_Labeled_Data.json
                
  
  3)TrainsModel: This file contains code for training SVM model with different balancing strategies.First you should choose any data from previous step.Then Three     things need to be initiated. 
  
                -groupby_col. Two options chould be selected for this value: "eventid" and "event_type" . It Specifies how to split training and test data.
                -sampling_strategy . It specifies sampling strategy. It could be:
                
         
                        -"none": imbalanced data
                        -"up" :upsampling
                        -"down": downsampling
                        -'up-with-same-eventtype':  when up-sampling, resample primarily from events of the same type
                        -'up-with-same-eventCategory': same event-type with the highest weight, data of the same “kind” of event (manmade vs. natural) weighted~6, and annotated data of other “kinds” of events weighted ~3)

                -up_weighting:could be true or false. It could be true for sampling_strategy : "none","up" and "down".If true, it is up-weighting samples from the same event type as the held-out event ( Directly re-weight samples)
   At the end , it generates the proper file name according to the parameters. Also, it is worth mentioning that the result only contains CrisisNLP and TREC IS       data.
