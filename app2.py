import streamlit as st
import tensorflow as tf
import pickle
import gdown
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def carrega_modelo():
  # https://drive.google.com/file/d/1vCvzakWS3EqmPJTRmEHxsyryvnWFRVs9/view?usp=sharing
  url = 'https://drive.google.com/uc?id=1vCvzakWS3EqmPJTRmEHxsyryvnWFRVs9'
  gdown.download(url,'modelo_vidente.keras')
  # https://drive.google.com/file/d/1HZ5e1X4whC8PIcPMOcJX43x55_Li03kI/view?usp=sharing
  url = 'https://drive.google.com/uc?id=1HZ5e1X4whC8PIcPMOcJX43x55_Li03kI'
  gdown.download(url,'vectorizer.pkl')
  loaded_model = tf.keras.models.load_model('modelo_vidente.keras')
  with open('vectorizer.pkl','rb') as file:
    vectorizer = pickle.load(file)
  return loaded_model, vectorizer

def predict_next_word(model, vectorizer, text, max_sequence_len, top_k=3):
    tokenized_text = vectorizer([text])
    tokenized_text = np.squeeze(tokenized_text)
    padded_text = pad_sequences([tokenized_text], maxlen=max_sequence_len, padding='pre')
    predicted_probs = model.predict(padded_text, verbose=0)[0]
    top_k_indices = np.argsort(predicted_probs)[-top_k::][::-1]
    predicted_words = [vectorizer.get_vocabulary()[index] for index in top_k_indices]
    return predicted_words

def main():
  max_sequence_len=50

  # Carregar Modelo
  loaded_model, vectorizer = carrega_modelo()

  st.title('Previsão de próximas palavras no texto informado')
  input_text = st.text_input('Digite uma sequencia de texto:')
  if st.button('Prever'):
    if input_text:
      try:
        predicted_words = predict_next_word(loaded_model, vectorizer, input_text, max_sequence_len)
        st.info('Palavras mais prováveis')
        for word in predicted_words:
          st.success(word)
      except:
        st.error('Erro na previsão {e}')
    else:
      st.warning('Por favor, insira algum texto;')

if __name__=='__main__':
  main()
    
