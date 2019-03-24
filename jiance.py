# coding:utf-8
import json
import codecs
import os
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# file = 'SQL+XSS.json'
data = []
label = []
print("##")
#数据读取，
with codecs.open('SQL+XSS.json','r','utf-8') as f:
	for line in f:
		dic = json.loads(line.strip())
		data.append(dic['request'])
		label.append('0')
print(len(data))
print(len(label))

with codecs.open('good-xss-200000.txt','r','utf-8') as f:
	for line in f:
		# dic = json.loads(line.strip())
		data.append(line)
		label.append('1')


print("#####")
print(len(data))
print(len(label))
data = data[:40000]
label = label[:40000]
data_train = data[:35000]
label_train = label[:35000]
label_test = label[35000:]
data_test = data[35000:]

#取出测试集，


# data1_train = ''.join(data_train)
# data1_test = ''.join(data_test)
# #将文本中的词语转换为词频矩阵
# vectorizer = CountVectorizer()
# #计算个词语出现的次数
# X = vectorizer.fit_transform(data)
# #获取词袋中所有文本关键词
# word = vectorizer.get_feature_names()


# count_vect = CountVectorizer()
# data1_train = [data1_train]
# data1_test = [data1_test]
# X_train_counts = count_vect.fit_transform(data1_train)
# X_test_counts  = count_vect.fit_transform(data1_test)


# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)

# weight_train = X_train_tf.toarray()
# print(weight_train.shape)

# tf_transformer = TfidfTransformer(use_idf=False).fit(X_test_counts)
# X_test_tf = tf_transformer.transform(X_test_counts)

vectorizer = CountVectorizer()#sklean中的创建一个向量计数器对象，CountVectorizer类会将文本中的词语转换为词频矩阵
tfidftransformer = TfidfTransformer()# TfidfTransformer用于统计vectorizer中每个词语的TF-IDF值
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(data_train))
weight = tfidf.toarray()
print(tfidf.shape)


#如果需要同时进行词频统计并计算TF-IDF值，则使用核心代码，transform()函数返回的矩阵的储存形式是稀疏矩阵。 
#可以用toarray()函数得到一个ndarray类型的完整矩阵。CountVectorizer就是一个支持文本中词语计数的函数库，我们可以使用其中的函数来分析文本数据来得到特征向量字典，
#字典中每一个项目的值代表该词语在全部数据中出现的次数
#在上述的代码中，我们首先使用了fit(..)方法来处理原始文本数据然后使用transform(..)方法来将词汇统计数据转换为tf-idf模型。这两部其实可以合并到一起以节省计算过程，
#我们可以使用如下所示的fit_transform(..)方法来实现这一点。词频的计算使用的是sklearn的TfidfVectorizer。这个类继承于CountVectorizer，在后者基本的词频统计基础上增加了如TF-IDF之类的功能。
#sklearn TfidfVectorizer，CountVectorizer，TfidfTransformer
#accuracy_score：分类准确率分数是指所有分类正确的百分比。分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。
#召回率 =提取出的正确信息条数 /样本中的信息条数。通俗地说，就是所有准确的条目有多少被检索出来了。


test_tfidf = tfidftransformer.transform(vectorizer.transform(data_test))
test_weight = test_tfidf.toarray()
print(test_weight.shape)


# # #查看特征结果
# # print(X_train_tf.shape)
# # print(X_train_tf.toarray())
# # print(X_train_counts.toarray())

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
	# fit the training dataset on the classifier
	classifier.fit(feature_vector_train, label)

	# predict the labels on validation dataset
	predictions = classifier.predict(feature_vector_valid)

	if is_neural_net:
		predictions = predictions.argmax(axis=-1)

	print("#########")


	return accuracy_score(predictions, label_test)



print(len(label_train))
print(len(label_test))
accuracy = train_model(MultinomialNB(), weight, label_train, test_weight)
print("NB, WordLevel TF-IDF: ", accuracy)

#success#################
# dtrain = MultinomialNB()
# dtrain.fit(weight,label_train)
# print(dtrain.score(test_weight,label_test))

# dtest = MultinomialNB(test_weight)  # label可以不要，此处需要是为了测试效果
# param = {'max_depth':6, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':11}  # 参数
# evallist  = [(dtrain,'train')]  # 这步可以不要，用于测试效果
# num_round = 100  # 循环次数
# bst = xgb.train(param, dtrain, num_round, evallist)
# preds = bst.predict(dtest)

# data1 = ''.join(data)

# model = TfidfModel(data1)
