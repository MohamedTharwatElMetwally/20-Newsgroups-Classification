# 20-Newsgroups-Classification
The 20 Newsgroups dataset is a collection of about 20,000 documents from 20 different newsgroups, covering various topics such as politics, religion, and sport. the task is building a model to classify news data into various categories through text classification.

There are three versions of the data set :-
- The first (19997 documents) is the original, unmodified version. 
- The second ("bydate", 18846 documents) is sorted by date into training(60%) and test(40%) sets, does not include cross-posts (duplicates) and does not include newsgroup-identifying headers (Xref, Newsgroups, Path, Followup-To, Date). 
- The third ("18828") does not include cross-posts (duplicates) and includes only the "From" and "Subject" headers.

the recommend dataset is the "bydate" version since cross-experiment comparison is easier (no randomness in train/test set selection), newsgroup-identifying information has been removed and it's more realistic because the train and test sets are separated in time.

Further Reading: http://qwone.com/~jason/20Newsgroups/