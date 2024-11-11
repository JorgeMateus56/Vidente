import streamlit as st
import gdown

@st.cache_resource
def carrega_modelo():
  url = 'https://drive.google.com/file/d/1vCvzakWS3EqmPJTRmEHxsyryvnWFRVs9/view?usp=sharing'
  gdown.download(url,'modelo_vidente.keras')
  url = 'https://drive.google.com/file/d/1HZ5e1X4whC8PIcPMOcJX43x55_Li03kI/view?usp=sharing'
  gdown.download(url,'vectorizer.pkl')
  loaded_model = tf.keras.models.load_model('modelo_vidente.keras')
  with open('vectorizer.pkl','rb') as file:
    vectorizer = pickle.load(file)
  return loaded_model, vectorizer

def main():
  max_sequence_len=50

  # Carregar Modelo
  loaded_model, vectorizer = carrega_modelo()

  st.title('Previsão de próximas palavras no texto informado')
  input_text = st.text_input('Digite uma sequencia de texto:')

if __name__=='__main__':
  main()
    
