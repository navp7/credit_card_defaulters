EXAMPLE:


Who survived the MS Estonia?
The MS Estonia sank in the Baltic Sea in 1994. Unfortunately, there were only 137 survivors of the 989 people on board. Although tragic, there exists a question of what led those 137 to survive. Did it have to do with their age, gender, or what type of passenger they were? Since shipwrecks are typically self contained, (the dataset provides the necessary data needed to make a prediction), this python script explores how different machine learning algorithms predict the fates of the passengers and crew.

Exploring and Cleaning the Data:

Dataset Balance

Something interesting to note is that this binary label (outcome) is extremely unbalanced, with around 87% of the people not surviving. This proved to be a challenge full of compromises when picking a model. You may see other projects with this dataset achieve high accuracy with simple algorithms, like linear regression. This is because a simple algorithm tends to underfit based on my observations. For example, if the model choose that everyone died, it would be 87% accurate on this dataset. In order to truly understand the effectiveness of one's model, they need to take precision, recall, and Fscore into consideration as well. In addition to that, I tuned sklearn's "class_weight" parameter to the balance of my dataset.

Age vs Survival Correlation:

The age vs survival plot seems to suggest that children are more likely to survive. However outside of that, there don't seem to be any distinct correlations.
Age vs Survival Correlation

Country vs Survival Correlation:

The country vs survival plot doesn't seem to suggest any correlations. On first glance it appears that German and English passengers have a better chance of surviving, but the error magrin is much larger, so I wouldn't be willing to make any hyptoheses based on them. Country vs Survival Correlation

Men (0) vs Women (1) Survival Correlation:

The gender plot is very revealing. There is an obvious correlation between being a male and survival. This is intersting because shipwrecks tend to see higher survival for women because of the way people are funneled into lifeboats (Women and children first). My hypothesis is that the gender correlation is intertwined with the passenger vs crew correlation because more of the crew members were men. However, this turned out not to be the case because the gender breakdown of the crew was almost a 50-50 split. Men vs Women Survival Correlation

Passenger vs Crew Survival Correlation:

The passenger vs crew plot also suggested that there was a correlation between being a crew member and survival. This also goes against the standard of seeing crew members let passengers board first. Passenger vs Crew Survival Correlation

Training Algorithms

Since part of the goal of this project was to learn about new algorithms, I decided to choose a broad sample of algorithms. I selected Logistic Regression, Support Vector Machines, Multilayer Perceptron, Random Forest, Decision Tree, Gradient Boosted Trees, and Histogram Gradient Boosted Trees. For each algorithm, I also used multilpe different parameters and cross validation to further distinguish the best models. I created functions to help display the outputs of each of the models. In order to use them later on, I pickled the best model of each algorithm to be used on the validation set.

Scoring and Evaluating Models

One of the most difficult decisions was how to score the models. Accuracy would be the obvious choice, however it does not tell the whole story. As mentioned in the Dataset Balance section, scoring based on accuracy alone is not ideal because it causes a very simple model to be successful without actually having to make any predictions. In particular, I saw more complex models favoring shallow trees, suggesting that the models thought that the correlations could be predicted with a very simple way. It turns out that the precision associated with these models was quite poor.

This left me questioning what exactly I should use to evaluate a model's success. I trained using precision, recall, and F1-score. When training for recall, I was able to achieve high recall at the expense of precision. This would have been fine, except I found that models that performed very well in recall tended to score low in accuracy. I believe that this is due to the inbalance in the dataset as well. Recall is equivalent to TP / TP + FN, where TP is true positive and FN is false negative. The likelihood of a false negative occuring is low becasue the data is already heavily skewed in a negative manner. Over the course of training, tuning to the least amount of false negatives drove up the likelihood of a false positive because the model wasn't being punished for making false positive predictions. Keeping in mind that precision is defined as a TP / TP + FP, where TP is true positive and FN is false negative, it makes sense that a model tuned for high recall would have high precision.

Visualizing PR Curves To visualize some of the thoughts I have discussed, I created precision-recall graphs based on models that were tuned for accuracy, precsision, and F-Score. Each of them is listed below with relevant commentary. On the whole, logistic regression, SVM, and mp were unable to make relevant predictions so I will not be discussing them.

PR Graph when training for accuracy

Across all of the algorithms, training for accuracy provided the smallest values for both precision and recall. As I mentioned earlier, I believe that this is due to the model underfitting across the board.
PR graph when trained for accuracy

PR Graph when training for precision PR graph when trained for precision

PR Graph when training for F-Score

Tuning for weighted F1-score optimized the models better than any of the other training methods. This came as no surprise because I was looking for a balance between recall and precision in my final model choice, but still needed to address the weighting issue. When tuning the the models for pure F1-score, the results were drastically worse. As with most of the problems I ran into, this was due to the class weighting. PR graph when trained for F-Score

Choosing the winner

Because of the inbalance in the dataset, GBT greatly benefited from its ability to learn from its mistakes when classifying data. Both of the gradient boosted models seemed to fare much better than other algorithms. Since the best model for this dataset seems to be a medium level complexity, the shallow tree depth did not seem to inhibit the performance of the model.

Concluding Scores, Gradient Boosted Trees:

Validation: Accuracy: 89.9% Precision: 81.3% Recall: 21.6% F1-Score: 34.2%

Test: Accuracy: 83.8% Precision: 45.7% Recall: 20.8% F1-Score: 28.6%

ML_Estonia/README.md at master · Akettle44/ML_Estonia 