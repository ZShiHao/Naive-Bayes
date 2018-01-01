#  Naive Bayes
# Filename: bayes-V1.1.py
import csv
import random
import math
from collections import Counter
from termcolor import colored

def loadcsv(filename):
	'''读取数据集'''
	lines=csv.reader(open(filename,'r'))
	dataset=list(lines)
	for i in range(len(dataset)):
		for s in [0,2,4,10,11,12]:
			dataset[i][s]=float(dataset[i][s])
	return dataset
	
def splitDataset(dataset,ratio):
	'''一部分数据集用于训练模型，一部分数据集用来测试'''
	trainSize=len(dataset)*ratio
	trainSet=[]
	copy=dataset.copy()
	while len(trainSet)<trainSize:
		index=random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	testSet=copy.copy()
	return [trainSet,testSet]

def separateClass(trainSet):
	'''将各实例映射到各类别中'''
	separated={}
	for i in range(len(trainSet)):
		instance=trainSet[i]
		if (instance[-1] not in separated):
			separated[instance[-1]]=[]
		separated[instance[-1]].append(instance)
	return separated

def handleDataset(dataset):
	'''删除有缺损的数据'''
	handledData=[]
	y=0
	for i in dataset:
		for x in range(len(dataset[0])):
			if i[x]=='?':
				y+=1
				break
			elif x==(len(dataset[0])-1):
				handledData.append(i)
	return handledData

def sumAttribute(separated):
	'''将各个类中的每个属性的值组成其自己的集合'''
	sumA={}
	attribute=[]
	for key in separated:
		sumA[key]=[]
		for x in range(len(separated[key][0])-1):
			for i in separated[key]:
				attribute.append(i[x])
			sumA[key].append(attribute.copy())
			attribute.clear()
	return [sumA,len(separated[key][0])-1]

def mean(numbers):
	'''计算每个属性的均值'''
	agv=sum(numbers)/float(len(numbers))
	return agv

def stdev(numbers):
	'''计算每个属性的标准差'''
	avg=mean(numbers)
	q=[]
	for i in numbers:
		q.append(pow(i-avg,2))
	SD=math.sqrt(sum(q)/float(len(numbers)-1))
	return SD

def maths(sumA,n):
	'''n为属性的个数
		计算对应类别中的实例的各个属性的特征值:①类别属性即其出现的概率 ②连续属性即标准差和均值(高斯分布)'''
	for key in sumA:
		for x in range(n):
			if x in [0,2,4,10,11,12]:
				c=sumA[key][x].copy()
				sumA[key][x].clear()
				sumA[key][x].append(mean(c))
				sumA[key][x].append(stdev(c))
			else:
				c=sumA[key][x].copy()
				t=Counter(c)
				u={}
				for attribute in t:
					u[attribute]=t[attribute]/len(sumA[key][x])
				e=u.copy()
				sumA[key][x].clear()
				sumA[key][x].append(e)
	sumM=sumA.copy()
	return sumM

def calculatePriorP(trainSet,separated):
	'''计算每种类别的先验概率'''
	PriorPs={}
	for key in separated:
		PriorPs[key]=(len(separated[key])/len(trainSet))
	return PriorPs

def calculateP(x,parameter):
	'''计算连续属性的类条件概率(根据高斯分布)'''
	agv=parameter[0]
	s=parameter[1]
	exponent=math.exp(-(math.pow(x-agv,2)/(2*math.pow(s,2))))
	Pi=(1/(math.sqrt(2*math.pi)*s))*exponent
	return Pi

def calculatePosteriorP(sumM,instance,n,PriorPs):
	'''计算后验概率'''
	Probabilities={}
	for key in sumM:
		p,pl=1,1
		for x in range(n):
			if x in [0,2,4,10,11,12]:
				pl*=calculateP(instance[x],sumM[key][x])
			else:
				for attribute in sumM[key][x]:
					if instance[x]==attribute:
						p*=sumM[key][x][attribute]
		Probabilities[key]=p*pl*PriorPs[key]
	return Probabilities

def predict(sumM,instance,n,PriorPs):
	'''预测单个实例的类别'''
	Probabilities=calculatePosteriorP(sumM,instance,n,PriorPs)
	bestLabel,bestProp=None,-1
	for classValue in Probabilities:
		if Probabilities[classValue]>bestProp:
			bestProp=Probabilities[classValue]
			bestLabel=classValue
	return bestLabel

def getPredictions(sumM,testSet,n,PriorPs):
	predictions=[]
	for instance in testSet:
		c=predict(sumM,instance,n,PriorPs)
		predictions.append(c)
	return predictions

def getAccuracy(testSet,predictions):
	correct=0
	for x in range(len(testSet)):
		if testSet[x][-1]==predictions[x]:
			correct += 1
	Accuracy=(correct/float(len(testSet)))*100.0
	return Accuracy

def trainModel(trainSet):
	'''训练数据集，得到模型:
		①数据的属性个数n
		②连续属性在其类别条件下的高斯分布，类别属性的类条件概率
		③类别的先验概率
		'''
	separated=separateClass(trainSet)
	PriorPs=calculatePriorP(trainSet,separated)
	sumA,n=sumAttribute(separated)
	sumM=maths(sumA,n)
	return [sumM,PriorPs,n]

def Test(testSet,sumM,PriorPs,n):
	predictions=getPredictions(sumM,testSet,n,PriorPs)
	Accuracy=getAccuracy(testSet,predictions)
	return Accuracy

def Data(filename,ratio):
	dataset=loadcsv(filename)
	handledData=handleDataset(dataset)
	trainSet,testSet=splitDataset(handledData,ratio)
	return [trainSet,testSet]

#读取数据集,数据预处理
filename='/users/zhangshihao/project/adult0.csv'
filename0='/users/zhangshihao/project/test.csv'
ratio=1
trainSet,testSet1 = Data(filename,ratio)
testSet,trainSet1=Data(filename0,0.7)
trainSet.extend(testSet)
trainSetN=len(trainSet)
testSetN=len(testSet)
#训练数据，得到分类器的模型
sumM,PriorPs,n = trainModel(trainSet)
#多重预测,测试模型
Accuracy = Test(testSet,sumM,PriorPs,n)
print(colored('The number of trainSet and testSet is:','blue',None,['bold']),end='')
print(colored(' {},{}'.format(trainSetN,testSetN),'blue'))
print(colored('The Accuracy is:','blue',None,['bold']),end='')
print(colored(' {}'.format(Accuracy),'blue'))
print(colored('The prior probabilities are:','blue',None,['bold']),end='')
print(colored(' {}'.format(PriorPs),'blue'))
print(colored('The parameter of each attribute:','blue',None,['bold']))
a=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
for key in sumM:
	print(colored("Class: {}".format(key),'red',None,['bold']))
	for x in range(n):
		print(colored('  {}:'.format(a[x]),'magenta',None,['bold']),end='')
		print(sumM[key][x])











