# StanfordSportsNER

This is a Named Entity Recognition (NER) model for identifying named entities within the sports domain. I have trained on NFL game recap 
articles retrieved from cbssports.com. In the initial commit (11/18/18), game recap articles of the first 9 weeks of the 2018 NFL season 
serve as my training set, and game recap articles from Week 10 of the season serve as my test set. I pre-process by tokenizing the articles 
using the program spacy_tokenize.py, which is found in my other repo, Sports_NER. In the Sports_NER repo, I attempted to build and train a 
similar NER model for the sports domain, using python's spaCy. I was disappointed with the perofrmance of this, however, and found 
StanfordCoreNLP's CRF-based NER model much more effective and better to work with. The CRF model takes roughly 35 minutes to train with 
the current NER feature specifications provided in propfile.txt. All data was labeled personally by me.  Training files used 
arelocated in the folder 2018_nfl_regular_season. Test files are in the folder unseen_files. Stanford's NER model accepts a file in TSV 
format, under a specific token-label structure, line by line, tab-separated. The fully labeled training set is in Sports_NER.tsv, and the test set is in unseen_files.tsv. Performance is very high, 0.9990 on training and 0.9907 on the test set.

Despite the high overall accuracy, there are some named entity classes that do not perform well. All of these are outlined in the results 
files for training and testing. It is mostly NER classes that are sparsely populated throughout the data that struggle with recall, and 
often get mislabeled as the default no-tag class (marked by a capital letter "O" as per the convention in StanfordCoreNLP's NER models). 
This is more pronounced in the test results than in the training results. I plan to get more data as the NFL season continues, as more 
game recap articles will be posted, and I will tokenize and label more data. Hopefully more data will increase the amount of sparse labels
enough to give better recall for these NER classes. There will always be mentions of rare named entity classes, however. There are also 
classes already present in the class set that are sort of a "catch-all", and the named entities that fall within this class can look very 
different from one another. One example is the REGION class, in which I basically attempt to capture those entities that indicate a 
geographical location, but are not cities, states, or countries, which have their own labels. Additionally in the sports domain, it is 
very common for the name of a city or state to be a reference to a TEAM, rather than a CITY. Determining whether or not a city name is a 
mention of a team, or if it refers to the actual city, presents one of the biggest challenges to NER modeling within this domain.

My ultimate goal in this project is to be able to tag any sports-related body of text accurately using this model. This might prove to be 
difficult if it's only trained on articles from one sport. For example, I tried to run a test on some baseball data using the model, and it 
could not accurately identify the names of AWARDs or PLAYOFF_ROUNDs, which all have different names in baseball. So using articles from 
multiple sports would obviously help with this endeavor. Tweaking the parameters in my properties file may help too, and I plan to do more 
research into StanfordCoreNLP's NERFeatureFactory to see what insights I can find.

Hope you enjoy this project, I enjoyed every minute of building it!
