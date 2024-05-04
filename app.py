import streamlit as st
import pickle
import nltk
tfid = pickle.load(open('tf.pkl','rb'))
Model = pickle.load(open('model.pkl','rb'))


from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):

    #lower the words
    text = text.lower()

    #tokenizing words
    text = nltk.word_tokenize(text)


#remove special charcters
    y = []
    for i in text:
        if i.isalnum:
            y.append(i)


    text = y[:]
    y.clear()


    #reoving stopwords and punctations
    for i in text:
       if i not in stopwords.words('english') and i not in string.punctuation:
           y.append(i)


    #stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS")


if st.button("Predict"):
    result = 3
    try:

        try:
             #1st preprocess the text
            processed_text = transform_text(input_sms)
            st.text("done processing")
        except Exception as e:
             st.error(f"An error in processing occurred: {str(e)}")


        try:
            #2nd vectorized txt
            vector_text = tfid.transform([processed_text])
            st.text("done vectorrization")
        except Exception as e:
            st.error(f"An error in vectorrization occurred: {str(e)}")
        
    
        try:
         #model
            result = Model.predict(vector_text)[0]
            st.text("result done")

        except Exception as e:
            st.error(f"An error result occurred: {str(e)}")

   
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    if result == 1:
        st.header("this is Spam")
    else:
        st.header("this is not spam")

    








# if not result:
#     st.error('Enter the text')
# else:
#     prediction = result[0]
#     confidence = mnbModel.predict_proba(vector_text).max()
#     if prediction == 1:
#         st.success(f"This is Spam ({confidence:.2f} confidence)")
#     else:
#         st.success(f"This is not Spam ({confidence:.2f} confidence)")



