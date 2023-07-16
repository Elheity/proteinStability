import streamlit as st
import pandas as pd 
import numpy as np 
import hashlib
#import the reqired libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re
import plotly.express as px
import pickle

train_data = pd.read_csv("/home/ahmedelheity/Downloads/Streamlit-master/data/train_data.csv")
training = train_data
training=training.dropna(how='all')
training['pH'] = training['pH'].fillna(training['pH'].mean())
training.drop_duplicates(subset=['protein_sequence','pH','data_source'],inplace=True)
training = training.drop(['data_source'],axis=1)

amino_count = training['protein_sequence'].str.split('').explode().value_counts().drop('')

# Protein Sequence Length as a column
training["protein_length"] = training["protein_sequence"].apply(lambda x: len(x))

# PhysioChemical Properties of Amino acids

#Aromaticity
def calculate_aromaticity(row):
  sequence = str(row[0])
  invalid_chars = re.findall("[^ACDEFGHIKLMNPQRSTVWY]", sequence)
  if invalid_chars:
      st.error(f"Invalid characters found in protein sequence: {invalid_chars}")
  else:
    X = ProteinAnalysis(sequence)
    return "%0.2f" % X.aromaticity()

#Molecular Weight
def calculate_molecular_weight(row):
  sequence = str(row[0])
  # Check for invalid characters in protein sequence
  invalid_chars = re.findall("[^ACDEFGHIKLMNPQRSTVWY]", sequence)
  if invalid_chars:
      st.error(f"Invalid characters found in protein sequence: {invalid_chars}")
  else:
      # Proceed with the calculation
      X = ProteinAnalysis(sequence)
      return "%0.2f" % X.molecular_weight()

#Instability Index
def calculate_instability_index(row):
  sequence = str(row[0])
  invalid_chars = re.findall("[^ACDEFGHIKLMNPQRSTVWY]", sequence)
  if invalid_chars:
      st.error(f"Invalid characters found in protein sequence: {invalid_chars}")
  else:
    X = ProteinAnalysis(sequence)
    return "%0.2f" % X.instability_index()

#Hydrophobicity
def calculate_hydrophobicity(row):
  sequence = str(row[0])
  invalid_chars = re.findall("[^ACDEFGHIKLMNPQRSTVWY]", sequence)
  if invalid_chars:
      st.error(f"Invalid characters found in protein sequence: {invalid_chars}")
  else:
    X = ProteinAnalysis(sequence)
    return "%0.2f" % X.gravy(scale='KyteDoolitle')

#Isoelectric Point
def calculate_isoelectric_point(row):
  sequence = str(row[0])
  print("sequence",sequence)
  invalid_chars = re.findall("[^ACDEFGHIKLMNPQRSTVWY]", sequence)
  if invalid_chars:
      st.error(f"Invalid characters found in protein sequence: {invalid_chars}")
  else:
    X = ProteinAnalysis(sequence)
    return "%0.2f" % X.isoelectric_point()

#Charge
def calculate_charge(row):
  sequence = str(row[0])
  invalid_chars = re.findall("[^ACDEFGHIKLMNPQRSTVWY]", sequence)
  if invalid_chars:
      st.error(f"Invalid characters found in protein sequence: {invalid_chars}")
  else:
    X = ProteinAnalysis(sequence)
    return "%0.2f" % X.charge_at_pH(row[1])
# Define the hashing function
def hash_func(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % 10**8

def return_amino_acid_df(df):
              # Feature Engineering on Train Data
              amino_acids=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
              for amino_acid in amino_acids:
                  df[amino_acid]=df['protein_sequence'].str.count(amino_acid,re.I)/df['protein_length']
                  #df[amino_acid]=df['protein_sequence'].str.count(amino_acid,re.I)
              return df



# Load the saved model from file
filename = 'data/model.pkl'
model1 = pickle.load(open(filename, 'rb'))



st.title("Protein Stability Predictor")
st.image("data//1.jpeg",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","About"])

if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(training.head(20))
    
    graph = st.selectbox("What kind of Graph ? ",["tm","Amino acid count"])

    #val = st.slider("Filter data using years",7)
    #data = train_data.loc[train_data["pH"]<= val]
    if graph == "tm":
        #plt.figure(figsize = (10,5))
        #plt.scatter(train_data["tm"],train_data["pH"])
        #plt.ylim(0)
        #plt.xlabel("pH")
        #plt.ylabel("tm")
        #plt.tight_layout()
        #st.pyplot()
        fig= px.histogram(training, x='tm', title="tm Distribution", template='plotly_dark',width=800,height=400)
        st.plotly_chart(fig)
    if graph == "Amino acid count":
        fig = px.bar(amino_count, x=amino_count.index, y='protein_sequence', color=amino_count.index)
        fig.update_layout(
            title='Amino Acid Count',
            height=600,
            template='ggplot2'
        )
        st.plotly_chart(fig)
  
if nav == "Prediction":
    st.header("Measure the enzyme stability")
    val1 = st.text_input('Enter the Protein Sequence', '')
    val2 = st.number_input("Enter the  PH",0.00,10.00,step = 0.25)
    val3 = st.text_input('Enter the protein source', '')
    # Check for invalid characters in protein sequence
    invalid_chars = re.findall("[^ACDEFGHIKLMNPQRSTVWY]", val1)
    if invalid_chars:
        st.error(f"Invalid characters found in protein sequence: {invalid_chars}")
    else:
        # Proceed with creating the DataFrame
        test_data = pd.DataFrame({
            'protein_sequence' : [val1],
            'pH' : [val2],
            'data_source' : [val3],
        })
    
        
        if not test_data.empty:
          print("None",test_data)
          print("Info",test_data.info())
          #print("Info",test_data['pH'])
          #test_data=test_data.dropna(how='all')
          #test_data['pH'] = test_data['pH'].fillna(test_data['pH'].mean())
          #test_data.drop_duplicates(subset=['protein_sequence','pH','data_source'],inplace=True)
          test_data = test_data.drop(['data_source'],axis=1)

          amino_count = test_data['protein_sequence'].str.split('').explode().value_counts().drop('')

          # Protein Sequence Length as a column
          test_data["protein_length"] = test_data["protein_sequence"].apply(lambda x: len(x))

          

          test_data = return_amino_acid_df(test_data)
          print("shape", test_data.shape)
          #test_data["protein_sequence"] = test_data["protein_sequence"].apply(hash_func)
          #test_data["data_source"] = test_data["data_source"].apply(hash_func)
          test_data['Aromaticity'] = test_data.apply(calculate_aromaticity, axis=1)
          test_data['Molecular Weight'] = test_data.apply(calculate_molecular_weight, axis=1)
          test_data['Instability Index'] = test_data.apply(calculate_instability_index, axis=1)
          test_data['Hydrophobicity'] = test_data.apply(calculate_hydrophobicity, axis=1)
          test_data['Isoelectric Point'] = test_data.apply(calculate_isoelectric_point, axis=1)
          test_data['Charge'] = test_data.apply(calculate_charge, axis=1)

          test_data.drop(columns=["protein_length"], inplace=True)
          test_data.drop(columns=["protein_sequence"], inplace=True)

          # Reset the DataFrame indexes
          test_data.reset_index(drop=True, inplace=True)


          test_data['Aromaticity'] = pd.to_numeric(test_data['Aromaticity'])
          test_data['Molecular Weight'] = pd.to_numeric(test_data['Molecular Weight'])
          test_data['Instability Index'] = pd.to_numeric(test_data['Instability Index'])
          test_data['Hydrophobicity'] = pd.to_numeric(test_data['Hydrophobicity'])
          test_data['Isoelectric Point'] = pd.to_numeric(test_data['Isoelectric Point'])
          test_data['Charge'] = pd.to_numeric(test_data['Charge'])

          Bestfeatures= ['pH', 'C', 'D', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Molecular Weight', 'Hydrophobicity', 'Isoelectric Point', 'Charge']
          #test_data["protein_sequence"] = test_data["protein_sequence"].apply(hash_func)
          #test_data["data_source"] = test_data["data_source"].apply(hash_func)
          #val = np.array(val).reshape(1,-1)
          if st.checkbox("Show Table"):
              st.table(test_data.head(20))

          pred =model1.predict(test_data[Bestfeatures])

          if st.button("Predict"):
              
              st.success(f"Your predicted tm is {pred}")

if nav == "About":
    st.title("About our model")
    st.text("The process of understanding the effect of pH on proteins ")
    st.text("and its potential contribution to mutations")

