# Classifying-EEG-signals-by-extracting-features-from-a-moving-time-window-using-different-ML-models
In this project I have extarcted 30 time and frequancy features from EEG signals (of left hand and right hand moving) in an espicific time window.
Then using PCA i have decreased the features dimension to 10.
Then I have quarried different methdos of ML: KNN(1,3,5,6), SVM(Linear kernel, Gaussian kernel), LDA, Naive bayes on different time windows.
I have used 2 different methods to validate the accuracy: Normal validation (train:70%, test:30%), Leave One Out 
Then I have found the richest time window for EEG signals and the dedicated ML model.
The highest accuracy belonged to the Gaussian SVM, almost 90%
