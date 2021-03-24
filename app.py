from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.externals import joblib
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
#train = pd.read_csv(r'C:\Users\Ranjeet shrivastav\Videos\ML_algorithms\Identify_the_Sentiments\train.csv')
#
#X=train['tweet']
#y = train['label']
## Extract Feature With CountVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer()
#X = cv.fit_transform(X) # Fit the Data
#pickle.dump(cv, open('tranform.pkl', 'wb'))
#
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(X,y,random_state=1,test_size=0.3)
#
#from xgboost import XGBClassifier
#xgb = XGBClassifier(max_depth = 6, n_estimators=1000)
#xgb.fit(x_train,y_train)
#
#filename='nlp.model.pkl'
#pickle.dump(xgb,open(filename,'wb'))

	if request.method == 'POST':
		message = request.form['tweet']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)