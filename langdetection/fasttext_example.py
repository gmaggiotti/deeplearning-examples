import fasttext
model = fasttext.load_model('lid.176.ftz')
print(model.predict('buenos dias',k=3))