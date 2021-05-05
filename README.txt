Aim: To classify cooking times for public recipes found on Food.com. We will be looking to test multiple different models, and combinations of models.

ImportData: Reads given features of the supplied data files and returns a dictionary with the data sets and types indexed.

SVMv1: This model uses support vector classification. "The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples... The multiclass support is handled according to a one-vs-one scheme." -https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

SVMv2: This model uses Stochastic Gradient Descent, do the the same thing as SVMv1. It underperformed in comparison, but was significantly faster. Scaling the vectors reduced performance as expected. Linear SVC was also tried. Performance was better than SGD but worse than SVM SVC.

Logistic Regression: This file contains 3 models that uses n_ingr, n_steps and both.

KNN: Applies k-Nearest-Neighbours classification. In its current form this is applied using n_ingr and n_steps as features and produces a plot of accuracy vs k.
