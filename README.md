# Twitter US Airline Sentiment Classifier (Natural Language Processing #2)
> You can find the dataset <a href="https://www.kaggle.com/crowdflower/twitter-airline-sentiment/version/2#">here</a>

## Objective
To classify tweets made by travelers in February 2015 as Neutral, Positive or Negative.

# Random Forest
I used the random forest classifier as the problem dealt with a relatively large dataset. Random Forests are also great classifiers when it comes to dealing with a large number of features.

#### Using max_features = 1600 and n_estimators = 550
<img src="cm.JPG" alt=""></img> 
### I got an accuracy of 75.4% (2208/2928)
However, is there a way to get better results? 

# Artificial Neural Networks 
I decided to use an ANN with:
#### 2 hidden layers (adding a third did not have significant effect in this case)
Overall, the ANN resulted in better classifications with an accuracy ranging between 
### Accuracy ~ 76-77%

## Conclusion
Finally, the results were not bad for the given dataset which contained many ambiguous/abbreviated tweets that would be difficult for a machine to interpret.

## Walking through the Code
### Random Forest
The steps taken were as follows:
<ul>
  <li>Get the Dataset</li>
  <li>Pre-process the text</li>
  <li>Create the Bag of Words Model</li>
  <li>Label Encode and OneHot Encode the Dependent Variable</li>
  <li>Split the data into Test and Training sets</li>
  <li>Train the Random Forest Classifier</li>
  <li>Get the Predicted values of test set</li>
  <li>Compare the predicted and test values and use a confusion matrix to calculate the accuracy of the model. </li>
  <li>Accuracy = (number of correct predictions on testing data / total number of testing data) </li>
</ul>
  
 ### ANN
The steps taken were as follows:
<ul>
  <li>Get the Dataset</li>
  <li>Pre-process the text</li>
  <li>Create the Bag of Words Model</li>
  <li>Label Encode and OneHot Encode the Dependent Variable</li>
  <li>Split the data into Test and Training sets</li>
  <li>Add Layers to your ANN</li>
  <li>Compile the ANN</li>
  <li>Get the Predicted values of test set</li>
  <li>Compare the predicted and test values and use a confusion matrix to calculate the accuracy of the model. </li>
  <li>Accuracy = (number of correct predictions on testing data / total number of testing data) </li> 
</ul>
