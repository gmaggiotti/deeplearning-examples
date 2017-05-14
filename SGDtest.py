from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.],[5,5]]
y = [0, 1,3]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
# SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
#               eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#               learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
#               penalty='l2', power_t=0.5, random_state=None, shuffle=True,
#               verbose=0, warm_start=False)

r = clf.predict([[1., 1.]])
print r


