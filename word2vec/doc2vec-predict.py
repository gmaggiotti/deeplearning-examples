# -*- coding: utf-8 -*-
# for word econding
import gensim
import MySQLdb

db = MySQLdb.connect(host="localhost",  # your host
                     user="root",       # username
                     passwd="bla",     # password
                     db="newscrawler")   # name of the database
#loading the model
model = gensim.models.doc2vec.Doc2Vec.load('valentin.model')

# #start testing
# #printing the vector of document at index 1 in docLabels
# docvec = model.docvecs[1]
#
# url = "/2144255-apoyo-del-fmi-al-cambio-en-el-bcra-dar-confianza-al-mercado-y-una-puerta-entreabierta-a-las-retenciones"
# sql = "select link, hash, date from article where link like '%" + url + "'"
#
#
# # Create a Cursor object to execute queries.
# cur = db.cursor()
# # Select data from table using SQL query.
# cur.execute(sql)
# # print the first and second columns
# for row in cur.fetchall() :
#     print row[0], " ", row[2]
#
# print
#to get most similar document with similarity scores using document- name
sims = model.docvecs.most_similar("0113.txt")
print sims

# ###  Get related links
# sql = " select link, date from article where "
#
# for item in sims:
#     if (item[1] > 0.80) :
#          sql += "hash='" + item[0] + "' or "
# sql += "0"
#
# # Create a Cursor object to execute queries.
# cur = db.cursor()
# # Select data from table using SQL query.
# cur.execute(sql)
# # print the first and second columns
# for row in cur.fetchall() :
#     print row[0], " ", row[1]
