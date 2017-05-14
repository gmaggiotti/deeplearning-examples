from sklearn.linear_model import LogisticRegression
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
clf.fit(X, y)
# SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
#               eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#               learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
#               penalty='l2', power_t=0.5, random_state=None, shuffle=True,
#               verbose=0, warm_start=False)

r = clf.predict([[0., .3]])
print r


