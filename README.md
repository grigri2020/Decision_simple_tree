# Decision_simple_tree

Data was obtained from Kaggle: https://www.kaggle.com/loveall/cervical-cancer-risk-classification

Although it was a risk identifying dataset, my goal was to see what are the features that make it possible to identity if the patient will have cervical cancer or not:
As expected HPV is the most important factor: root. Given below is the tree created:



<img src=output/Decision_Tree_cervical.png> 


test_size is set to 0.25 which is the defeault in DecisionTreeClassifier(). 
>>The default criterion for node split is "gini". 
But this can be changed to  
>>DecisionTreeClassifier(criterion = 'entropy')
