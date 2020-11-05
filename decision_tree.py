
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

"""
LOAD DATA 
"""

clean = np.loadtxt('./wifi_db/clean_dataset.txt')
noisy = np.loadtxt('./wifi_db/noisy_dataset.txt')


"""

DECISION TREE LEARNING

"""

"""
Decision Tree Model Structure:
The root node will be passed to functions as the "model"

Non-leaf node structure:
  {'attribute': attribute for split, 
   'value':     value of the attribute to split, 
   'left':      left branch from this node, 
   'right':     right branch from this node}

Leaf node structure:
  {'attribute': attribute for split, 
   'value':     value of the attribute to split, 
   'left':      left branch from this node, 
   'right':     right branch from this node,
   'label':     predicted label for data that reaches this node}

"""

#Returns root node of the resulting decision tree and the max depth
def decision_tree_learning(training_dataset, depth = 1):
	#training_dataset: Dataset as matrix, with label as last column
	#depth: Used to compute the maximal depth of the tree, for plotting purposes for instance
	if all_samples_have_same_label(training_dataset):
		return {'attribute': 0, 'value': 0,'left': None, 'right': None, 'label': training_dataset[0][-1]}, depth 
	else:
		split = find_split(training_dataset)
		l_branch, l_depth = decision_tree_learning(split['l_dataset'], depth + 1)
		r_branch, r_depth = decision_tree_learning(split['r_dataset'], depth + 1)
		node = {'attribute': split['attribute'], 'value': split['value'], 'left': l_branch, 'right': r_branch}
	return node, max(l_depth, r_depth)

#Returns true or false for given dataset
def all_samples_have_same_label(dataset):
	label1 = dataset[0][-1]
	for sample in dataset:
		label2 = int(sample[-1])
		if label1 != label2:
			return False
	return True

#Returns the predicted label for data based on the given decision tree model
def predict(model, data):
	#model: root node of the decision tree model
	#data: data instance to be predicted a label
	current_node = model
	if is_leaf(current_node):
		return current_node['label']
	else:
		attribute = current_node['attribute']
		value = current_node['value']

	if data[attribute] > value:
		return predict(current_node['right'], data)
	else:
		return predict(current_node['left'], data)

#Returns true if the node is a leaf node, returns false otherwise
def is_leaf(node):
	if node['left'] == None and node['right'] == None:
		return True
	else:
		return False

"""

FIND SPLIT

"""

"""
Find the split point with highest info gain
dataset    - data that needs to be split into two subsets
best_split - dictionary that contains:
	attribute - used to make the split
	value     - the unique value to make the split
	l_dataset - the data that went to the left based on the split
	r_dataset - the data that went to the right based on the split
"""

#Returns dictionary of the resulting best split
def find_split(dataset):
	#init
	best_split = {'attribute': 0, 'value': 0, 'l_dataset': 0, 'r_dataset': 0}
	final_gain = 0
	r, c = dataset.shape

	#if length of thr dataset is 0 returns empty dictionary
	if r == 0:
		return best_split   
	H_all = get_H(dataset) #H for overal dataset
	
	#runs a loop through all attributes
	for attr_idx in range(c - 1):
		#depending on the attribute used sorts the data
		data_sort = dataset[np.argsort(dataset[:, attr_idx])]

		#creates a set of unique values for that attribute
		val_unique = set(data_sort[:, attr_idx])
		
		#loops through the set of unique values
		for val in val_unique:
			#initiates the arrays 
			l_data = np.empty((0, c))
			r_data = np.empty((0, c))
	 		#goes row by row
			for row in data_sort:
				#if the value of the attribute column is above unique -> right
				if row[attr_idx] > val:
					r_data = np.vstack((r_data, row))
		 		#if the value of the attribute column is below unique -> left
				else:
					l_data = np.vstack((l_data, row))

			#calculates the gain basedon the devided data
			gain = H_all - get_remainder(l_data, r_data)

			#if the newly calculated gain is better than the previous best one -> replace it
			if gain > final_gain:
				final_gain = gain
				best_split = {'attribute': attr_idx, 'value': val, 'l_dataset': l_data, 'r_dataset': r_data}

	return best_split

#Returns entropy of the data
def get_H(dataset):
	#dataset: recieves dataset 
	#H: returns entropy values for the given dataset

  labels = dataset[:, -1] #looks at the labels of the dataset
  _, count_all = np.unique(labels, return_counts = True) #counts how many times each label appears (in the form of an array)
  p = count_all/count_all.sum() #calculates the probability
  H = -1 * sum(p * np.log2(p))  #calculates H
  return H

#Returns remainder 
def get_remainder(l_data, r_data):
	#l_data: recives the data that goes to the left branch
	#r_data: recieves the data that goes to the right branch
	#remainder: returns the remainder calcualtions 

	H_right = get_H(r_data) 
	H_left = get_H(l_data)
	
	S_right = len(r_data) #length of the left data sample 
	S_left = len(l_data)  #length of the right data sample
	
	remainder = H_left * (S_left/(S_left + S_right)) + H_right * (S_right/(S_left + S_right))
	return remainder

"""

PERFORMANCE MEASURMENTS/EVALUATION

"""
#Calculate all metrics and pretty print
#Returns printed evaluation metrics and average precision, recall, f1, class_rate
def pretty_evaluate (confusion_matrix, trees, name):
	#confusion_matrix: the calculated confusion matrix, int [4][4]
	#Get precision, recall, f1, classification_rate (accuracy)
	precision, recall = get_precision_recall (confusion_matrix)
	f1 = [get_f1 (precision[i], recall[i]) for i in range (4)]
	class_rate = get_classification_rate (confusion_matrix)
	depth = find_depth(trees[1])

	#Print values
	print('Confusion matrix:')
	print('Actual class: \t\t(1) \t(2) \t(3) \t(4)')
	for i in range(4):
		print('Predicted class (%i):\t'%(i+1), round(confusion_matrix[i][0],0),'\t', round(confusion_matrix[i][1],0),'\t',round(confusion_matrix[i][2],0),'\t', round(confusion_matrix[i][3],0))
	print()

	plot_matrix(confusion_matrix, name)
	print()

	print('Class \t | Precision \t\t| Recall \t\t| F1')
	for i in range(4):
		print('(%i)\t |'%i, round(precision[i],4), '\t\t|', round(recall[i],4), '\t\t|', round(f1[i],4))
	print()
	print('Average classification rate:', class_rate)
	print()
	print('Depth of the final tree:', depth)
    
	return precision, recall, f1, class_rate, depth

#Precision and recall
#Returns precision: float[4], recall: float[4]
def get_precision_recall(confusion_matrix):
	#confusion_matrix: the calculated confusion matrix

	#Initialize arrays for precision and recall metrics
	#where the position in the array (index+1) indicates the class
	precision, recall = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0] 
	#Loop through dataset once for each class (4 times overall)
	for i in range(0,4):
		tp = confusion_matrix[i][i]    # True Positives
		p = sum(confusion_matrix[i])  # All Positives
		fp = p - tp                   # False Positives
		fn = sum([confusion_matrix[k][i] for k in range(0,4)]) - tp # False Negatives
	#Calculate precision and recall values for current class
		if tp != 0:
			precision[i] = tp/p
			recall[i]= tp/(tp+fn)
	#Return precision and recall arrays
	return precision, recall

#F1 Score
#Returns f1: float
def get_f1 (precision, recall):
	#precision: calculated precision
	#recall: calculated recall
	if precision+recall == 0:
		return 0
	return 2*(precision*recall)/(precision+recall)

#Accuracy/ Classification rate
#Returns classification_rate: float
def get_classification_rate (confusion_matrix):
	#confusion_matrix: the calculated confusion matrix

	#Initialize counters for correct and all values
	correct, total = 0, 0
	#Loop through dataset, once for each class (4 times overall)
	for i in range(0,4):
		correct += confusion_matrix[i][i]    # True Positives
		total += sum(confusion_matrix[i])      # Everything else
	#Make resilient against empty datasets
	if total==0:
		return 0
	return correct/total

#Evaluate tree
#(Alternative to get_classification_rate)
def evaluate(tree_node, dataset):
	#tree_node: node
	#dataset:  array (with last column as class label)
	classifier = lambda test_data: predict(tree_node, test_data)
	return get_classification_rate(get_confusion_matrix (classifier, dataset))


#Confusion matrix
#Returns confusion_matrix: int[4][4]
def get_confusion_matrix (tree, dataset):
	#tree: tree classifier as function
	#dataset: data as array (with last column as class label)

	#Initialize confusion matrix conf_matrix[i][j]
	#where i+1 indicates the predicted class
	#and j+1 indicates the true class
	conf_matrix = [[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0]]
	#Loop through elements in dataset
	for row in dataset:
		#Determine true and predicted value
		true = int(row[-1])
		pred = int(tree(row[0:-1]))
		#Update confusion matrix
		conf_matrix[pred-1][true-1] += 1

	#Return confusion matrix
	return conf_matrix

#Alternatively parametrised version for get_confusion_matrix
#Retursn consufion_matrix: int [4][4]
def get_confusion_matrix_tree (tree_node, dataset):
	#tree_node: node
	#dataset:  array (with last column as class label)
	classifier = lambda test_data: predict(tree_node, test_data)
	return get_confusion_matrix(classifier, dataset)

#Calculates the depth of the tree
#Returns tree'sdepth
def find_depth(node):
    if is_leaf(node):
      return 1;
    else:
      return 1 + max(find_depth(node['left']), find_depth(node['right']))

"""

PRUNNING

"""

"""
Non-leaf node structure:
  {'attribute': attribute for split, 
   'value':     value of the attribute to split, 
   'left':      left branch from this node, 
   'right':     right branch from this node}

Leaf node structure:
  {'attribute': attribute for split, 
   'value':     value of the attribute to split, 
   'left':      left branch from this node, 
   'right':     right branch from this node,
   'label':     predicted label for data that reaches this node}
"""

#Prunes the given tree (in node format) based on given dataset
#Returns pruned tree
def prune (tree_node, dataset):
	#tree_node: the node of the tree in form of dict
	#dataset: data

	#Get children
	left_node = tree_node['left']
	right_node = tree_node['right']

	#Split dataset according to tree
	left_data = []
	right_data = []
	for element in dataset:
		if element[tree_node['attribute']] > tree_node['value']:
			right_data.append(element)
		else:
			left_data.append(element)
	#Check whether both children are leaf nodes
	if is_leaf(left_node) and is_leaf(right_node):
	#Find majority class
		if len(right_data) > len(left_data):
			majority = right_node['label']
		else:
			majority = left_node['label']
		#Make tree_node leaf node
		new_node = {'attribute': 0, 'value': 0,'left': None, 'right': None, 'label': majority}
		#Evaluate old and new sub-trees
		if evaluate(new_node, dataset) >= evaluate(tree_node, dataset):
			return new_node
		else:
			return tree_node
	#Check whether only left node is leaf node
	elif is_leaf(tree_node['left']):
		new_node = tree_node.copy()
		new_node['right']=prune (right_node, right_data)
		if is_leaf(new_node['right']):
			return prune(new_node, dataset)
		else: 
			return new_node
	#Check whether only right node is leaf node
	elif is_leaf(tree_node['right']):
		new_node = tree_node.copy()
		new_node['left']=prune (left_node, left_data)
		if is_leaf(new_node['left']):
			return prune(new_node, dataset)
		else: 
			return new_node
	#Neither one is leaf node
	else:
		new_node = tree_node.copy()
		new_node['right']=prune (right_node, right_data)
		new_node['left']=prune (left_node, left_data)
		if is_leaf(new_node['right']) and is_leaf(new_node['left']):
			return prune(new_node, dataset)
		else: 
			return new_node

"""

VALIDATION

"""

"""
STEP 1: The method train_test_split splits the data into a training set and a test set. 
The method takes the dataset and the size of the test set as input (float value between 0 and 1, a common value would be 0.2).
""" 
#Return the train and the test set
def train_test_split(dataset, test_size_percent):
	#dataset: data to be seperated
	#test_size_percent: what percentage of data to be tested on (i.e. 0.2)
    #Compute the number of rows in the test set.
    test_size_absolute = int(test_size_percent*len(dataset))
    #Shuffle the dataset to ensure a random train-test-split.
    np.random.shuffle(dataset)
    #Generate a test and a train set in accordance with the specified ratio (test_size_percent)
    test = dataset[:test_size_absolute, :]
    train = dataset[test_size_absolute:, :]
    
    #Return the train and the test set
    return train, test

"""
STEP 2: Devide the dataset into k folds for cross-validation
"""
def k_fold(k, dataset):
	#k: number of folds for cross-valid
	#dataset: data to be seperated
    #Shuffle again (We can delete this step of we don't need it)
    np.random.shuffle(dataset)
    
    #Split the dataset into k near-equally sized groups and save them in an array.
    k_subsets = np.array_split(dataset, k, axis=0)
    
    #Return an array of k near-equally sized subsets of the train set which can be used for cross-validation
    return k_subsets

"""
CROSS-VALIDATION
"""
#Validate the performance of the algorithm by training and evaluating it on k folds. Optional: Compare the performance of a pruned and unpruned tree. 
def cross_validation(k, dataset, prune_tree=False):
	#k: number of folds for cross-valid
	#dataset: data to be seperated
	#prune_tree: boolean of whether tree needs pruning (Flase by default)
    #Generate k folds for cross_validation
    k_subsets_test = k_fold(k, dataset)

    #Initiate lists to store the models resulting from the k training iterations
    number_pruned_trees = k * (k-1)
    trees = [None] * k
    trees_prune = [None] * number_pruned_trees
    
    #Initiate a (k,4,4) dimensional numpy array to store the confusion matrices observed in the k evaluation iterations (numpy array to simplify arithmetic operations)
    matrix = np.empty((k, 4,4))
    matrix_prune = np.empty((number_pruned_trees, 4,4))

    print("")
    #loop through the k subsets to train a model and compute the accuracy for each of them
    for i in range(0, k):
        #Train-Test-Split per fold
        test_data = k_subsets_test[i]
        train_data = np.delete(k_subsets_test, i, 0)
        
        print("Iteration (", i+1, "/", k, ")")
        #Nested cross-validation to evaluate the performance of the pruned model 
        if (prune_tree):
            for j in range(0, k-1):
                val = train_data[j]
                train = np.delete(train_data, j, 0)
                train = train.reshape((train.shape[1]*(k-2),8))
                #Train the model on the training data
                
                trees_prune[j+i*(k-1)],_ = decision_tree_learning(train) 
                #Prune the tree using the validation set
                trees_prune[(j+i*(k-1))] = prune(trees_prune[(j+i*(k-1))], val)
                #Calculate the confusion matrix of the pruned tree
                confusion_matrix = get_confusion_matrix_tree(trees_prune[(j+i*(k-1))], test_data)
                matrix_prune[(j+i*(k-1))] = np.array([confusion_matrix]) 
            
    
        #Train the model on the training data
        train_data = train_data.reshape((train_data.shape[1]*(k-1),8))
        trees[i],_ = decision_tree_learning(train_data)
        #Calculate the confusion matrix of the unpruned model
        confusion_matrix = get_confusion_matrix_tree(trees[i], test_data)
        matrix[i] = np.array([confusion_matrix])  
    
    #Average the confusion matrix over all k unpruned trees
    average_matrix = matrix.sum(axis=0)/k

    #If the algorithm is initialized with prune_tree = True, it returns the models and performance for both the unpruned and pruned versions.
    if (prune_tree):
        #Average the confusion matrix over all k pruned trees
        average_matrix_prune = matrix_prune.sum(axis=0)/(k*(k-1))
        print("")
        print("---------------------------")
        print("Results for the unpruned tree") 
        print("---------------------------")
        pretty_evaluate(average_matrix, trees, name)
        print("")
        print("---------------------------")
        print("Results for the pruned tree") 
        print("---------------------------")
        pretty_evaluate(average_matrix_prune, trees_prune, 'cm_prune')
        return trees, trees_prune, average_matrix, average_matrix_prune
    else: 
        print("")
        print("---------------------------")
        print("Results for the unpruned tree") 
        print("---------------------------")
        pretty_evaluate(average_matrix,trees, 'cm_unprune')
        return trees, average_matrix

"""
VISUALIZATION
"""

#Visualises the tree 
def visualize_tree(tree, depth, name):
  figure, axes = plt.subplots(figsize=(50, 10))
  dy = 1/depth	
  #calls for visualise node function 
  visualize_node(tree, 0, 1, 0, 1, dy, axes)
  plt.show()
  plt.savefig(name+".png")


#Visualises a node of the tree
def visualize_node(node, xmin, xmax, ymin, ymax, dy, axes):
  annotation = 'att:' + str(node['attribute']) + '\nval:' + str(node['value'])

  midx = (xmax-xmin)/2 + xmin
  dx = (midx-xmin)/2
  #if node is a leaf no lines out, just the box
  if is_leaf(node):
    axes.annotate('label: '+str(node['label']), xy=(midx, ymax), xycoords="data", va="center", ha="center", 
                  bbox=dict(boxstyle="round", fc="w"))
  #Node left then visualise the node, and have lines going to the next nodes
  if node['left'] != None:
    axes.annotate(annotation, xy=(midx-dx, ymax-dy), xytext=(midx, ymax), va="center", ha="center", 
                  bbox=dict(boxstyle="square", fc="w"), arrowprops=dict(arrowstyle="->"))
    visualize_node(node['left'], xmin, midx, ymin, ymax-dy, dy, axes)
  #Node left then visualise the node, and have lines going to the next nodes
  if node['right'] != None:
    axes.annotate(annotation, xy=(midx+dx, ymax-dy), xytext=(midx, ymax), va="center", ha="center", 
                  bbox=dict(boxstyle="square", fc="w"), arrowprops=dict(arrowstyle="->"))
    visualize_node(node['right'], midx, xmax, ymin, ymax-dy, dy, axes)

#Plot the confusion matrix with a heat map
def plot_matrix(confusion_matrix, name):
	#confusion matrix
	predicted = ["Predicted class (1)", "Predicted class (2)", "Predicted class (3)", "Predicted class (4)"]
	actual = ["Actual class (1)", "Actual class (2)", "Actual class (3)", "Actual class (4)"]
	
	confusion = np.round(confusion_matrix,1)

	fig, ax = plt.subplots()
	

	im = ax.imshow(confusion, cmap=cm.summer)
	
	#set x/y labels
	ax.set_xticks(np.arange(len(actual)))
	ax.set_yticks(np.arange(len(predicted)))
	ax.set_xticklabels(actual)
	ax.set_yticklabels(predicted)

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")
	#plot the confusion matrix values
	for i in range(len(actual)):
		for j in range(len(predicted)):
			text = ax.text(j, i, confusion[i, j],
				ha="center", va="center", color="black")

	#fig.tight_layout()
	plt.show()
	plt.savefig(name+".png")


"""

 MAIN

"""

#CLEAN DATA 
print("\nCLEAN DATASET:")
clean_trees, clean_trees_prune, clean_average_matrix, clean_average_matrix_prune = cross_validation(10, clean, True)

visualize_tree(clean_trees[1], find_depth(clean_trees[1]), "clean")
visualize_tree(clean_trees_prune[1], find_depth(clean_trees_prune[1]), "clean-pruned")


#NOISY DATA
print("\nNOISY DATASET:")
noisy_trees, noisy_trees_prune, noisy_average_matrix, noisy_average_matrix_prune = cross_validation(10, noisy, True)

visualize_tree(noisy_trees[1], find_depth(noisy_trees[1]), "noisy")
visualize_tree(noisy_trees_prune[1], find_depth(noisy_trees_prune[1]), "noisy-pruned")




