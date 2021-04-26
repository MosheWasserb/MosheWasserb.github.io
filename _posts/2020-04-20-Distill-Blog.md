| | | | |
|-|-|-|-|
|The unreasonable effectiveness ofdistillation - Text Classification  Part 1/3 | | | |
| | | | |
|In recent years,increasingly large Transformer-based models such as BERT, have demonstratedremarkable state-of-the-art (SoTA) performance in many Natural LanguageProcessing (NLP) tasks and have become the de-facto standard. | | | |
|However there is no free lunch (indeed? Wait you read this blog series), thosemodels are extremely inefficient and require massive computational resourcesand large amounts of data as basic requirements for training and deploying.This severely hinders the scalability and deployment of NLP-based systems across the industry.| | | |
| | | | |
|I have always beenfascinated by robustness and efficiency in NLP targeting production. So I decided to write a series of blogs that will providesome practical tips with code samples regarding how to deploy and adapt SoTAlarge transformer models. | | | |
| | | | |
|In the first 3blogs I will focus on model distillation for Text-Classification. Modeldistillation is a very powerful pruning technique and in many use-cases yieldssignificant speed-up and memory size reduction. But still considered more foradvanced users: relatively hard to implement, un-predictable performance,minimization of multi-loss errors,  and lack of understanding of theinherent mechanism. I will try to show the opposite by providing a few realuse-case examples with simple code snippets and distiiation?s efficacyintuition. | | | |
| | | | |
|Ok so let?sstart. | | | |
|Suppose that youare a data scientist in one of the top leading enterprise companies and yourtask is to classify social media tweets and deliver SoTA models intoproduction. You have also been told that your model must be very efficientsince you will pay a high fee for any extra milion parameters. So most likelyyou will start by collecting sufficient data (a few hundred/thousands oflabeled samples) and compare a few ML/DL models and Transformer models totarget high accuracy with minimum cost (model size). | | | |
| | | | |
| that consists of Twitter postslabeled with any of six basic emotion categories: sadness, disgust, anger, joy,surprise, and fear. The data consists of 16K training samples and 2K testsamples and available on HuggingFace data-set Hub. A code example for thefollowing steps are available here.| | | |
| | | | |
|1st  step: Set a baseline using alogistic regression model (TfIdf based)| | | |
| | | | |
|Accuracy = 86.1%| | | |
| | | | |
|Our baseline result. | | | |
| | | | |
|2nd step: Deep Learning baseline| | | |
|Next, we will try asimple MLP Neural Network (NN). The model architecture is very basic andincludes an input embedding size of 16, a word vocabulary size of 5000, and onehidden layer.| | | |
| | | | |
|Accuracy = 86%| | | |
|Number ofparameters = 80K | | | |
| | | | |
|Similar accuracy toLogistic regression but more efficient and dense. Weare veterans in the NLP domain and by now already practiced with HuggingFaceand their amazing ?transformers? library so let?s try a fewpopular SoTA transformer models. | | | |
| | | | |
|3rd: Transformers models| | | |
| | | | |
| Model| Accuracy | #Parameters | Implementation Source |
| BERT| 92.4 | 110M | Our |
| Roberta| 93 | 110M | Our |
| T5| 93.5 | 11B | |
| | | | |
| | | | |
|Awsome, accuracy fliesto the sky but ? 110M parameters are far above our computational budget,and for sure IT hits the ceiling when they hear about the 11B model (:| | | |
| | | | |
|So insteadlet?s try one of the more popular models like DistillBERT thatHugginngFace released, half the size and double the speed vs. BERT base.| | | |
| | | | |
|4th step: DistillBERT| | | |
| | | | |
| Model| Accuracy | #Parameters | Implementation Source |
| DistillBERT| 91.5 | 67M | Our |
| DistillRoberta | 92.3 | 67M | |
| | | | |
| | | | |
|Nice, we gained<1% loss vs. BERT model and a much smaller one. | | | |
| | | | |
|So that?s it?Are we done? Are we ready to go with this model to production and pay thecomputing cost for the 67M parameters?| | | |
| | | | |
|Can we do more? Canwe do with less than 1M or 100K with minimal accuracy loss (<1% loss)? | | | |
| | | | |
|The answer is?yes? (a spoiler: but in general the answer varys and depends on yourdataset quality and task in-hand) with the help of distillation and additionaldata either augmented from the training set or sampled from your in-domainunlabeled data-set. | | | |
| | | | |
|). | | | |
| | | | |
|) has shown promisingresults for reducing the model size and computational load while preservingmuch of the original model?s performance. A typical model distillationsetup includes two stages. In the first stage, a large, cumbersome and accurateteacher neural network is trained for a specific downstream task.  In the second stage, shown in Figure 1,a smaller and simpler student model that is more practical for deployment inenvironments with limited resources, is trained to mimic the behavior of theteacher model.| | | |
| | | | |
| | | | |
|Figure 1:  Student model training process. | | | |
| | | | |
|). The KL loss allows us touse the same code for standard training (labeled data) or distillation (pseudolabeled data). | | | |
| | | | |
|5th step: Distill Roberta to a simpler student| | | |
|In our caselet?s distill Roberta?s knowledge into our simple MLP NN. | | | |
|Following are theresults:| | | |
| | | | |
| Model| Accuracy | #Parameters | |
| MLP| 86 | 80K | |
| MLP_D| 91.8 | 80K | |
| | | | |
| | | | |
|Wow!!! Surprisinglynot bad at all, on-par accuracy with DistilBERT.  | | | |
| | | | |
|The followingfigure summarizes the results that we achieved so far for Emotion in terms of(model acc./BERT acc.)% Vs. model size.| | | |
| | | | |
| | | | |
|Figure: Unreasonable example for theeffectiveness of distillation: Accuracy of a distilled MLP student (80Kparameters) is on-par with BERT (the teacher with 110M params), BERT Large(330M params) and T5 (11B params) on Emotions dataset.| | | |
| | | | |
| | | | |
|Ok, We were able to distill Roberta?s knowledge into ourtiny model with almost no loss. | | | |
|We benefit fromhigh transformer model performance with very little cost to pay (and for ourIT).| | | |
| | | | |
|In this stage youprobably have many questions, here are few examples:| | | |
|?     Does this?trick? hold for any classification sub-task (or any otherNLP?s tasks) specifically on our own data-set? | | | |
|?     If not, whendoes it work? | | | |
|?     How to choosethe best student for my data-set?  | | | |
|?     What is theintuition behind this mechanism?| | | |
| | | | |
|I?ll try toanswer those questions and give a bit of intuition in the following blogs.   | | | |
| | | | |
| | | | |
| | | | |
| | | | |
|The unreasonable effectiveness ofdistillation - Text-Classification Part 2/3| | | |
| | | | |
|In the first blog, we achieved a surprisinglygood performance by distilling large (teacher) models into tiny (student)models that are on-par with the transformer mammoth models. In this blog, weintend to further explore it and verify whether these results can also beachieved for other text-classification?s data-sets and sub-tasks.| | | |
|)benchmark.| | | |
| | | | |
|SST-2 Dataset | | | |
|SST-2 The Stanford Sentiment Treebank 2comprises single sentences extracted from movie reviews and  binary (positive/negative)sentiment classification labels.| | | |
|, the data consists of 6920 trainingsamples and 1800 test samples.. | | | |
| | | | |
|Following are the results of few transformermodels, MLP and the distillation of the RoBERTa model into the MLP.| | | |
| | | | |
| Model| Accuracy | #Parameters | |
| RoBERTa (teacher)| 93 | 110M | |
| BERT-base| 91.4 | 110M | |
| DistillBERT | 90.4 | 66M | |
| MLP| 79 | 80K | |
| MLP-D| 86 | 80K | |
| | | | |
|Oops, we notice a drop of 6% with MLP-Dcompared to BERT-base.| | | |
|So, this time the tiny model fails to learnthe full capacity of knowledge that allows it to decode the classification taskas well as the teacher model.| | | |
|Instead, let's try to replace our studentmodel with a much deeper architecture but still significantly smaller vs. BERT:A  Bi-LSTMwith 1.7M parameters (x80).| | | |
| | | | |
|Following are the results of the Bi-LSTMstudent distilled model.| | | |
| | | | |
| Model| Accuracy | #Parameters | |
| Bi-LSTM-D| 90 | 1.7M | |
| | | | |
|Not bad, we only have a 2% accuracy drop andwe are on-par with DistillBERT.| | | |
|So, for SST-2, a simple Bi-LSTM student whouldbe considered as sufficient for production purposes.| | | |
| | | | |
| | | | |
|The followingfigure summarizes the results that we achieved so far for SST-2 in terms of(model acc./BERT acc.)% Vs. model size.| | | |
| | | | |
| | | | |
| | | | |
|TheCoLA Dataset and Task| | | |
| The Corpus of Linguistic Acceptabilityconsists of English acceptability judgments drawn from books and journalarticles on linguistic theory. Each sentence is associated with a label that indicateswhether it is a grammatical English sentence or not. | | | |
| | | | |
|Following are the results for the teachermodel (BERT) and the two student distilled models :| | | |
| | | | |
| Model| MCC | #Parameters | |
| BERT| 52.8 | 110M | |
| MLP+D| 10 | 80K | |
| Bi-LSTM+D| 24 | 1.7M | |
| | | | |
|Whoops, in contrast to the SST-2 dataset eventhe Bi-LSTM doesn?t hold the capacity to match BERT on the CoLAtask.  Both the Bi-LSTM and theMLP+Attn.models are far from being comparable to BERT. In the case of SST-2task, Bi-LSTM which is a larger and deeper model than MLP closed the gap withthe teacher model (BERT). | | | |
| | | | |
|Let?s try a model with a larger anddeeper model than Bi-LSTM for the CoLA task, but one that is still moreefficient compared to the teacher model.| | | |
|We chose DistilBERT, a very popular model byHuggingFace based on self-distill of logits, and embeddings. | | | |
| | | | |
| | | | |
|Following are the results:| | | |
| | | | |
| Model| MCC | #Parameters | |
| DistilBERT| 51.3 | 66M | |
| | | | |
|Great, now we only have 1.5% accuracy drop andwe are on-par with BERT base.| | | |
|So, for the CoLA task, a self-distilled Bertstudent will be considered as sufficient  for production purposes.| | | |
| | | | |
|The followingfigure summarizes the results that we have so far with CoLA. | | | |
| | | | |
| | | | |
| | | | |
|Note: At this point it may be trivial to think that Bi-LSTM/MLP didnt holdthe full capacity of BERT since they have simpler architecture but rememberthat KD dependent on the availability of un-labeled in-domain data. In the caseof CoLA we didnt have sufficient in-domain data so wehad to generate data using data different augmentation techniques.  ColA?s grammatical acceptence taskis very challenging so not sure we were able to generate high quality data thatrepresent the full task?s data distribution. This is very importantresearch direction, we have noticed significant improvement (2-3 points) whendistill with generative model (e.g. GPT-2) vs. Easy Data  Augmentatoion (EDA) which basedon semantic replacement of random key-words. | | | |
| | | | |
|SomeIntuition  | | | |
|In general we showedthat it is feasible to distil BERT using very efficient models while preservingcomparable results. However, the success of the distillation (student size vs.accuracy) depends on the dataset and task at hand. | | | |
|What is the reason for such variance inperformance?| | | |
|In order to answer this questionwe need first to look into the different datasets.| | | |
|Here are few data instances taken from the 3 dataset:| | | |
| | | | |
| Dataset| Instance examples | Category | Task |
| Emotion| im updating my blog because i feel shitty | Sadness | Emotion |
|  | i just feel greedy and lame making one | Anger |   |
|  | im feeling more comfortable with derby | Joy |   |
| SST-2| works because , for the most part , it avoids the stupid cliches and formulaic potholes that befall its brethren | POS | Sentiment |
|  | but the characters tend to be cliches whose lives are never fully explored | NEG |   |
| CoLA | many evidence was provided | unacceptable | Grammatical acceptability |
|  | The jeweller inscribed the ring with the name     | acceptable |   |
|  | The gardener planted roses in the garden | acceptable |   |
|  | The more books I ask to whom he will give, the more he reads | unacceptable |   |
| | | | |
|A successful classification of the Emotiontask seems to be heavily dependent on lexical cues - salient emotional wordsthat represent the emotional category regardless of structure and syntax. Onthe other hand the CoLA task is inherently dependenton the syntax structure and less on lexical cues. Whereas, successfulclassification of theSST-2 task seems to be dependent on a mix of lexical andsyntactic clues.| | | |
| | | | |
|So in order to successfully distill the full teacher knowledge which isrequired for a given task, the student architecture must be capable to absorbits teacher capacity that is related to the task. In the case of CoLA andpartially SST-2, the architecture of the MLP model does not have the capacityto learn the full syntactic information stored in the teacher.  This is clear since the MLP?sarchitecture is basically based on Bag Of Word (BOW)embedding implementation. Classification of the Emotion?s task requiresthe learning of mostly semantic lexical knowledge therefore MLP is sufficientand Bi-LSTM and DistillBERT are over-qualified,| | | |
|. | | | |
|To summarize: Classification tasks thatrequire capturing of general lexical semantics cues can be successfullydistilled by very simple and efficient models; however, classification tasksthat require the detection of linguistic structure and contextual relations aremore challenging for distillation using simple student models| | | |
| | | | |
|In the next blog, we will suggest a simplemetric to estimate the success of the distillation.| | | |
|We also showcase a simple dynamic architecturethat utilizes this metric to achieve optimal tradeoff between size and accuracyfor a given data set/task. Then we propose a few future researchdirections.    | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
|The unreasonable effectiveness ofdistillation - Word Order Sensetivity and Dynamic Data Aware Transformer  Part 3/3| | | |
| | | | |
|In the last two blogs we saw useful examplesof text classification distillation methods.| | | |
|We presented some intuition why distillationworks and how to choose the smallest student model that can match the capacityof its teacher model.| | | |
|In this blog, I would like to suggest a metricfor estimating the complexity level of your dataset/task and exploit it tooptimize the distillation performance. We?ll then propose a naive switchtransformer architecture in order to maximize the simplicity of ourdataset/task to the efficiency in terms of student model size. We will end theseries of blogs with a few thoughts on the transformer structure and futureresearch directions. | | | |
| | | | |
|Wordorder sensitivity| | | |
| showed a surprising phenomenon: between 75% and 90% ofthe correct predictions of Transformer-based classifiers, trained on GLUEtasks, remained unchanged when the input words were randomly shuffled. Theauthors further suggested a simple metric to measure dataset sensitivity toword order:| | | |
| | | | |
|WOS (Word Order Sensitivity) = (100-p)/50 | | | |
|where p is theaccuracy of a task-trained model evaluated on a dev-s set | | | |
|(See Thang?sSec 3.1 and 2.3.2)| | | |
| | | | |
|, that shows the WOS  scores plotted for various GLUEtasks followed by a table that presents our measure of BERT WOS score forEmotion, SST-2 and CoLA datasets (1-gram shuffling):| | | |
| | | | |
| | | | |
| | | | |
| | | | |
| Dataset| WOS | | |
| Emotion| 0.14 | | |
| SST-2| 0.34 | | |
| CoLA| 0.99 | | |
| | | | |
| | | | |
|Our intuition, discussed in the previous blog,was right! The CoLA dataset with average WOS score of 0.99 which means that itconsists of a vast majority of ?hard? samples where the Emotiondataset has the lowest WOS score which means that it consists of a vastmajority of ?simple? samples. | | | |
|The SST-2 WOS score is 0.34 which means that it  tends to havemore ?simple? instances than ?hard? ones. Those resultsare quite consistent with the distillation performances (student model size vs.accuracy): the Emotion dataset was successfully(almost without loss) distilled to a tiny MLP model, SST-2 to Bi-LSTM model andCoLA to DistillBERT model.  | | | |
| | | | |
|IMDB?sexample | | | |
|Lets apply our new metric to the popular IMDB dataset and try to predictthe distillation results. | | | |
|)comprises single sentences extracted from informal movie reviews for binary(positive/negative) sentiment classification.| | | |
|The training dataconsists, for this example, a subset of 1K randomly sampled samples from the25K training samples and 25K test samples.| | | |
| | | | |
|We have calculated the WOS score of IMDB to be0.28. Since WOS is relatively low (<0.3) we anticipate that an MLP orBi-LSTM models should be suitable for absorbing the capacity of its teachermodel (Roberta in this case).| | | |
| | | | |
|Here are the results of the distillation ofthe IMDB dataset/task :| | | |
| | | | |
| Model| Accuracy | #Model Parameters | |
| BERT| 88.6 | 110M | |
| DistilBERT| 87.7 | 110M | |
| RoBERTa (teacher)| 91.6 | 110M | |
| MLP | 80 | 80K | |
| MLP+D| 90 | 80K | |
| Bi-LSTM+D| 91 | 0.7M | |
| | | | |
| | | | |
|Indeed, as we predicted, a small MLP/Bi-LSTMmodel is capable of absorbing RoBERTa?s knowledge for the IMDBdataset/task and even outperforms BERT performance. | | | |
| | | | |
| | | | |
|TheSwitch Architecture| | | |
| proposed a transformerearly exit based on the confidence of the prediction in each layer, or in otherwords the transformer dynamically expands/shrinks its size during  inference based on the complexityof each inference sample. As we showed so far, a dataset with the majority ofsimple instances can be distilled by a very efficient model. So, continuing along thisline we suggest to apply a ?simple\hard? predictor of each inputinstance during inference time, and decide which student model to use based onthis prediction. We refer to it as ?the switch architecture?: forthe ?simple? instances use the student model and for the?hard? ones use a teacher model (no distillation). Since thestudent?s model architecture is considerably more efficient in relationto the teacher model (e.g. MLP/Bi-LSTM vs. BERT) for a dataset that consists ofa majority of simple instances theaverage speed-up boost will be high with the switch architecture| | | |
| | | | |
|Below a diagram that presents an abstract viewof the switch architecture and followed by simple/hard instance examples fromSST-2 dataset:| | | |
| | | | |
| | | | |
| | | | |
| Dataset| Simple Instance examples | Hard Instance examples | |
| SST-2| extremely confusing | the issue of faith is not explored very deeply | |
|  | the rock is aptly named | works because , for the most part , it avoids the stupid cliches and formulaic potholes that befall its brethren | |
|  | sturdy entertaining period drama both caine and fraser have their moments | what it lacks in originality it makes up for in intelligence and b grade stylishness | |
|  | moving and invigorating film | succeeds in providing a disquiet world the long dreaded completion of the police academy series | |
|  | an intelligent | but the characters tend to be cliches whose lives are never fully explored | |
|  | good movie | the story and characters are nowhere near gripping enough | |
| | | | |
| | | | |
|Following are some examples of thepotential speed-up gains that can be achieved for the SST-2 dataset/task withdifferent student models:| | | |
| | | | |
| Model| *Speed-Up vs. BERT-base | Accuracy | |
| RoBERTa (teacher)| x1 | 93.54 | |
| BERT-base| x1 | 91.4 | |
| DistillBERT| x2 | 90.4 | |
| Bi-LSTM| **x40 | 81 | |
| Bi-LSTM-D| x40 | 90 | |
| Switch Bi-LSTM-D+BERT| x20 | 91.4 | |
| 5% hard samples| x5 | 92.5 | |
| Switch Bi-LSTM-D+BERT| | | |
| 20% hard samples| | | |
|*In our code example the instance classifier(hard/simple) implementation is far from being efficient, we need to run BERTtwice on each test data instance. I?ll leave it for future research andbe happy to get ideas to improve it. In the table, the computing overhead for the classification has beenneglected.| | | |
|. | | | |
| | | | |
|Dataaware transformer and future research| | | |
|BERT holds extensive knowledge on languagestructure including semantic/syntactic cues. It well known that afterfine-tuning BERT utilizes partial and sufficient knowledge to solve a giventask on a specific domain.  Whatmaybe innovative contribution to BERTology, based on the switch arch resultsabove, it seems that during inference BERT ?retrive? sufficientknowledge to decode a specific instance, for simple examples shallowinformation (aka semantic) for hard examples higher level information (akasysntactic or even world knolwedge). As a result dataaware training allows to distill this knowledge into simpler architecture withlimited capacity. | | | |
|We encourage to continue and explore dataaware optimization techniques in order to | | | |
|dynamically adapt transformer size and speedin production. In addition, it would be important to understand when a datainstance is ?simple? or ?hard? for a given task andexploit this prediction for an early exit or efficient switch arch. Forchallenging tasks (majority of the data are ?hard? instances) like CoLA,SQuAD, OpenQA would be interesting to explore how external knowledge retrievalcould transform ?hard? instances to ?simple? ones togain maximum efficiency.   | | | |
| | | | |
| | | | |
