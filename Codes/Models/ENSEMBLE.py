# Ensemble for combining various models

import cPickle
import sys
import operator
import random
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,pos,l_dist,r_dist,entity_pos,ner,dep_d1,dep_d2,dep_path):
        self.entity_pair = entity_pair
        self.sentences = sentences 
        self.label = label
        self.pos_tags = pos
        self.ner = ner
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos
        self.dep_dist1 = dep_d1
        self.dep_dist2 =dep_d2
        self.dep_path_bw_en = dep_path

def read_data(file_name):
	prob_data = cPickle.load(open(file_name))
	# returning the probabilities for each class predicted, and the true label
	return (prob_data[3],prob_data[2])

class ensemble():

	def train(self,classifier_probabilities,true_y):
		# have a linear ensemble of classifiers which predicts the confirdence of each option separately and then choose the best option 
		X=[]
		y=[]
		# construct X , by flatenning the probabilities. therefore a1,a2,a3,a4 and b1,b2,b3,b4 with y are flattened to a1,b1,I(y[0]) .. a2,b2,I(y[1]).. etc.
		for i in range(len(classifier_probabilities[0])):
			tempy=[0.0]*len(classifier_probabilities[0][0])
			tempy[true_y[i][0]]=1.0
			y.extend(tempy)
			for j in range(len(classifier_probabilities[0][0])):
				temp=[]
				for k in range(len(classifier_probabilities)):
					temp.append(classifier_probabilities[k][i][j])
				X.append(temp)

		print len(y),len(X),len(X[0]),'.... linear regression data details'
		# linear model
		regr = linear_model.LinearRegression()
		regr.fit(np.array(X),np.array(y))
		print 'Coefficients: \n', regr.coef_," ", type(regr.coef_)
		return regr.coef_

	def test(self,test_prob_classifier,weights,true_y,entity_pair):
		for i in range(len(test_prob_classifier)):
			test_prob_classifier[i]=weights[i]*test_prob_classifier[i]
		y= (test_prob_classifier[0]+test_prob_classifier[1]+test_prob_classifier[2])/3
		test_pr = self.pr(y,true_y,entity_pair)
		return test_pr

	def equal_test(self,classifier1,classifier2,classifier3,true_y,entity_pair):
		y= (classifier1+classifier2+classifier3)/3
		test_pr = self.pr(y,true_y,entity_pair)
		return test_pr

	def pr(self,predict_y,true_y,entity_pair):
	    final_labels = []
	    for label in true_y:
	        if 0 in label and len(label) > 1:
	            label = [x for x in label if x!=0]
	        final_labels.append(label[:])

	    total = 0
	    for label in final_labels:
	        if 0 in label:
	            continue
	        else:
	            total += len(label)
	    print "Total:",total
	    results = []
	    for i in range(predict_y.shape[0]):
	        for j in range(1, predict_y.shape[1]):
	            results.append([i,j,predict_y[i][j],entity_pair[i]])
	    resultSorted = sorted(results, key=operator.itemgetter(2),reverse=True)
	    p_p = 0.0
	    p_n = 0.0
	    n_p = 0.0
	    pr = []
	    prec = 0.0
	    rec = 0.0
	    p_p_final = 0.0
	    p_n_final = 0.0
	    n_p_final = 0.0
	    prev = -1
	    for g,item in enumerate(resultSorted):
	        prev = item[2]
	        if 0 in final_labels[item[0]]:
	            if item[1] == 0:
	                continue
	            else:
	                n_p += 1
	        else:
	            if item[1] in final_labels[item[0]]:
	                p_p += 1
	            else:
	                p_n += 1

	        pr.append([(p_p)/(p_p+n_p+p_n), (p_p)/total])
	        if rec <= 0.3:
	            prec = (p_p)/(p_p+n_p+p_n)
	            rec = (p_p)/total
	            p_p_final = p_p
	            p_n_final = p_n
	            n_p_final = n_p

	    print "p_p:",p_p_final,"n_p:",n_p_final,"p_n:",p_n_final
	    return [prec,rec,total,pr]

if __name__=='__main__':

	dataset_path = sys.argv[1]
	# loading pre calculated probabilities of each class in the dev set
	classifier_1_dev_prob = read_data(dataset_path+"/dev/prob_dev_WA.pkl")
	classifier_2_dev_prob = read_data(dataset_path+"/dev/prob_dev_EA.pkl")
	classifier_3_dev_prob = read_data(dataset_path+"/dev/prob_dev_PCNN.pkl")

	# loading pre calculated probabilities of each class in the test set
	classifier_1_test_prob = read_data(dataset_path+"/test/prob_WA.pkl")
	classifier_2_test_prob = read_data(dataset_path+"/test/prob_EA.pkl")
	classifier_3_test_prob = read_data(dataset_path+"/test/prob_PCNN.pkl")

	# make entity_pairs
	test_data_bags = cPickle.load(open(dataset_path+"/preprocessed_dataset/test.pkl"))
	entity_pair=[data_bag.entity_pair for data_bag in test_data_bags]

	ensemble_classifier = ensemble()
	#test_pr = ensemble_classifier.equal_test(classifier_1_test_prob[0],classifier_2_test_prob[0],classifier_3_test_prob[0],classifier_1_test_prob[1],entity_pair)
	weights = ensemble_classifier.train([classifier_1_dev_prob[0],classifier_2_dev_prob[0],classifier_3_dev_prob[0]],classifier_1_dev_prob[1])
	test_pr = ensemble_classifier.test([classifier_1_test_prob[0],classifier_2_test_prob[0],classifier_3_test_prob[0]],weights,classifier_1_test_prob[1],entity_pair)

	out = open(dataset_path+"/pr/pr.txt",'w')
	for i in range(0,len(test_pr[3]),10):
		res = test_pr[3][i]
		out.write(str(res[0])+"\t"+str(res[1])+"\n")
	out.close()
