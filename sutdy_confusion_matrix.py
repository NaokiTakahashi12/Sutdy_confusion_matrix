
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

if __name__ == '__main__':
	iris = load_iris();

	print(iris.DESCR)

	x = iris.data
	y = iris.target

	pca = PCA(n_components=2)
	X2 = pca.fit_transform(x)

	plt.plot(X2[y==0,0], X2[y==0,1],"ro")
	plt.plot(X2[y==1,0], X2[y==1,1],"g.")
	plt.plot(X2[y==2,0], X2[y==2,1],"bx")

	x_scaled = preprocessing.scale(x)

	x_scaled.mean(axis=0),x_scaled.std(axis=0)

	clf = KNeighborsClassifier(n_neighbors=1)

	print(clf)

	scores = cross_val_score(clf, x_scaled, y, cv=10)
	y_pred = cross_val_predict(clf, x_scaled, y, cv=10)

	print(confusion_matrix(y, y_pred))
	is_confusion_matrix = confusion_matrix(y, y_pred);

	TP = is_confusion_matrix[0,0] + is_confusion_matrix[1,1] + is_confusion_matrix[2,2]
	FN = is_confusion_matrix[0,1] + is_confusion_matrix[1,2] + is_confusion_matrix[2,1]
	FP = is_confusion_matrix[1,0] + is_confusion_matrix[2,1] + is_confusion_matrix[1,2]

	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	F1 = 2*((precision*recall) / (precision + recall))

	print(" ------ Not use method ------")
	print("Precision is")
	print(precision)
	print("Recall is")
	print(recall)
	print("F1 is")
	print(F1)

	print(" ------ Use method ------")
	print("Precision is")
	print(precision_score(y, y_pred, average='macro'))
	print("Recall is")
	print(recall_score(y, y_pred, average='macro'))
	print("F1 is")
	print(f1_score(y, y_pred, average='macro'))

	plt.show()

