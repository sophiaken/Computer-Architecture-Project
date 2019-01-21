import time
import timeit
import tensorflow as tf
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

print("started")
import time
import timeit
import tensorflow as tf
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

print("started")

iris = datasets.load_iris()
X, y = iris.data, iris.target

X = np.repeat(X, 100, axis=0)
y = np.repeat(y, 100, axis=0)

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


NUM_THREADS = 10
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))


with tf.device('/gpu:0'):
  start = time.time()
  clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))
  clf.fit(X, y)
  end = time.time()
  print ("Single SVC gpu", end - start, clf.score(X,y))
  proba = clf.predict_proba(X)
  t=tf.convert_to_tensor(proba, dtype=tf.float32)

  n_estimators = 10
  start = time.time()
  clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators, bootstrap=False))
  clf.fit(X, y)
  end = time.time()
  print ("Bagging SVC gpu", end - start, clf.score(X,y))
  proba = clf.predict_proba(X)
  t=tf.convert_to_tensor(proba, dtype=tf.float32)
  


def gpu():
  sess.run()
  

sess.close()



    
   



