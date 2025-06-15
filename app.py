import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import streamlit_option_menu as som
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model_RFC.joblib')
severity_path = os.path.join(script_dir, 'Dataset', 'Symptom-severity.csv')

st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Script directory: {script_dir}")
st.sidebar.write(f"Model path: {model_path}")
st.sidebar.write(f"Model exists: {os.path.exists(model_path)}")
st.sidebar.write(f"Severity file exists: {os.path.exists(severity_path)}")

if os.path.exists(severity_path):
    severity = pd.read_csv(severity_path)
    severity['Symptom'] = severity['Symptom'].str.replace("_", " ")
    symptoms_list = severity['Symptom'].tolist()
    st.sidebar.success("✅ Severity data loaded successfully!")
else:
    st.error(f"Severity file not found at: {severity_path}")
    symptoms_list = []

# Check if model file exists and load it
if os.path.exists(model_path):
    model_RFC = jb.load(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.error(f"Model file not found at: {model_path}")
    st.info("Please train the model first using the 'Train Model' tab.")
    model_RFC = RandomForestClassifier(n_estimators=100, random_state=42)

# Sidebar menu
with st.sidebar:
    menu_option = ['Prediction', 'Add Data', 'Train Model']
    selected_option = som.option_menu(
        'Disease Prediction System Based on Symptoms',
        options=menu_option,
        icons=['hospital', 'database-fill-add', 'train-front'],
        menu_icon='bandaid'
    )

# Helper function to create symptom input with custom option first
def create_symptom_input(label, symptoms_list, key):
    # Create dropdown options with "Enter custom symptom" as first option
    dropdown_options = ['', '🖊️ Enter custom symptom'] + symptoms_list
    selected_option = st.selectbox(label, dropdown_options, index=0, key=f"select_{key}")
    
    # If user selects "Enter custom symptom", show text input
    if selected_option == '🖊️ Enter custom symptom':
        custom_symptom = st.text_input(f"✏️ Enter {label.lower()}:", key=f"custom_{key}", placeholder="Type your custom symptom here...")
        return custom_symptom.strip() if custom_symptom.strip() else None
    else:
        return selected_option if selected_option else None

# Prediction page
if selected_option == 'Prediction':
    if not os.path.exists(model_path):
        st.warning("⚠️ Model not found! Please train the model first using the 'Train Model' tab.")
        st.info(f"Looking for model at: {model_path}")
        st.stop()

    st.header('Disease Prediction System based on Symptoms')
    st.info("Select symptoms from dropdown or enter custom symptoms in the text fields")

    # Use dropdowns instead of text inputs for valid symptom selection
    col1, col2, col3 = st.columns(3)
    with col1:
        Sym_1 = st.selectbox('Symptom 1', [''] + symptoms_list, index=0)
        Sym_4 = st.selectbox('Symptom 4', [''] + symptoms_list, index=0)
        Sym_7 = st.selectbox('Symptom 7', [''] + symptoms_list, index=0)
        Sym_10 = st.selectbox('Symptom 10', [''] + symptoms_list, index=0)
        Sym_13 = st.selectbox('Symptom 13', [''] + symptoms_list, index=0)
        Sym_16 = st.selectbox('Symptom 16', [''] + symptoms_list, index=0)
    with col2:
        Sym_2 = st.selectbox('Symptom 2', [''] + symptoms_list, index=0)
        Sym_5 = st.selectbox('Symptom 5', [''] + symptoms_list, index=0)
        Sym_8 = st.selectbox('Symptom 8', [''] + symptoms_list, index=0)
        Sym_11 = st.selectbox('Symptom 11', [''] + symptoms_list, index=0)
        Sym_14 = st.selectbox('Symptom 14', [''] + symptoms_list, index=0)
        Sym_17 = st.selectbox('Symptom 17', [''] + symptoms_list, index=0)
    with col3:
        Sym_3 = st.selectbox('Symptom 3', [''] + symptoms_list, index=0)
        Sym_6 = st.selectbox('Symptom 6', [''] + symptoms_list, index=0)
        Sym_9 = st.selectbox('Symptom 9', [''] + symptoms_list, index=0)
        Sym_12 = st.selectbox('Symptom 12', [''] + symptoms_list, index=0)
        Sym_15 = st.selectbox('Symptom 15', [''] + symptoms_list, index=0)

    def prediction(Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17):
        # Input data
        data = [Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17]
        
        # Debug: Show input symptoms
        st.write(f"Input symptoms: {data}")
        
        # Load severity data
        severity = pd.read_csv(severity_path)
        severity['Symptom'] = severity['Symptom'].str.replace("_", " ")
        sym = np.array(severity['Symptom'])
        weight = np.array(severity['weight'])

        # Encode symptoms to weights
        for i in range(len(data)):
            if data[i] and data[i] in sym:
                data[i] = weight[np.where(sym == data[i])[0][0]]
            else:
                data[i] = 0

        # Debug: Show encoded input
        st.write(f"Encoded input: {data}")
        
        # Make prediction
        pred = model_RFC.predict([data])[0]
        
        # Get prediction probabilities
        prob = model_RFC.predict_proba([data])[0]
        classes = model_RFC.classes_
        prob_dict = {cls: prob[i] for i, cls in enumerate(classes)}
        
        # Debug: Show prediction probabilities
        st.write(f"Prediction probabilities: {prob_dict}")
        
        return pred

    if st.button('Make Prediction'):
        dia_prediction = prediction(Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17)
        st.success(f"Predicted Disease: {dia_prediction}")

# Add Data page
elif selected_option == 'Add Data':
    st.title('Your Contribution is Valuable!')
    st.write('##### Provide data here')
    st.info("💡 Select '🖊️ Enter custom symptom' from dropdown to add new symptoms, or choose from existing ones")
    
    label = st.text_input('Disease Label')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Sym_1 = create_symptom_input('Symptom 1', symptoms_list, 1)
        Sym_4 = create_symptom_input('Symptom 4', symptoms_list, 4)
        Sym_7 = create_symptom_input('Symptom 7', symptoms_list, 7)
        Sym_10 = create_symptom_input('Symptom 10', symptoms_list, 10)
        Sym_13 = create_symptom_input('Symptom 13', symptoms_list, 13)
        Sym_16 = create_symptom_input('Symptom 16', symptoms_list, 16)
    with col2:
        Sym_2 = create_symptom_input('Symptom 2', symptoms_list, 2)
        Sym_5 = create_symptom_input('Symptom 5', symptoms_list, 5)
        Sym_8 = create_symptom_input('Symptom 8', symptoms_list, 8)
        Sym_11 = create_symptom_input('Symptom 11', symptoms_list, 11)
        Sym_14 = create_symptom_input('Symptom 14', symptoms_list, 14)
        Sym_17 = create_symptom_input('Symptom 17', symptoms_list, 17)
    with col3:
        Sym_3 = create_symptom_input('Symptom 3', symptoms_list, 3)
        Sym_6 = create_symptom_input('Symptom 6', symptoms_list, 6)
        Sym_9 = create_symptom_input('Symptom 9', symptoms_list, 9)
        Sym_12 = create_symptom_input('Symptom 12', symptoms_list, 12)
        Sym_15 = create_symptom_input('Symptom 15', symptoms_list, 15)

    def add_new_symptom_to_severity(new_symptom, default_weight=1):
        """Add new symptom to severity CSV with default weight"""
        try:
            severity = pd.read_csv(severity_path)
            # Check if symptom already exists
            existing_symptoms = severity['Symptom'].str.replace("_", " ").str.lower().str.strip()
            if new_symptom.lower().strip() not in existing_symptoms.values:
                # Add new symptom with default weight
                new_row = pd.DataFrame({
                    'Symptom': [new_symptom.replace(" ", "_")],
                    'weight': [default_weight]
                })
                severity = pd.concat([severity, new_row], ignore_index=True)
                severity.to_csv(severity_path, index=False)
                st.success(f"✅ New symptom '{new_symptom}' added to severity database with weight {default_weight}")
                return True
        except Exception as e:
            st.error(f"Error adding new symptom to severity database: {str(e)}")
            return False
        return False

    def add_data(label, Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17):
        dataset_path = os.path.join(script_dir, 'Dataset', 'dataset.csv')
        if not os.path.exists(dataset_path):
            st.error(f"Dataset file not found at: {dataset_path}")
            return False
        if not os.path.exists(severity_path):
            st.error(f"Symptom severity file not found at: {severity_path}")
            return False

        # Collect all symptoms
        symptoms = [Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17]
        
        # Check for new symptoms and add them to severity database
        severity = pd.read_csv(severity_path)
        severity['Symptom'] = severity['Symptom'].str.replace("_", " ")
        existing_symptoms = severity['Symptom'].str.lower().str.strip().tolist()
        
        new_symptoms_added = []
        for symptom in symptoms:
            if symptom and symptom.strip():
                clean_symptom = symptom.lower().strip()
                if clean_symptom not in existing_symptoms:
                    if add_new_symptom_to_severity(symptom.strip()):
                        new_symptoms_added.append(symptom.strip())
                        existing_symptoms.append(clean_symptom)

        # Prepare data for insertion
        data = [label] + symptoms
        for i in range(1, len(data)):
            if data[i]:
                data[i] = str(data[i]).lower().strip()

        # Load updated severity data
        severity = pd.read_csv(severity_path)
        severity['Symptom'] = severity['Symptom'].str.replace("_", " ")
        
        # Add data to dataset
        try:
            dataset = pd.read_csv(dataset_path)
            df = pd.DataFrame([data], columns=dataset.columns)
            dataset = pd.concat([dataset, df], ignore_index=True)
            dataset.to_csv(dataset_path, mode='w', index=False)
            
            if new_symptoms_added:
                st.info(f"📝 New symptoms added: {', '.join(new_symptoms_added)}")
            
            return True
        except Exception as e:
            st.error(f"Error adding data to dataset: {str(e)}")
            return False

    # Show current input summary
    symptoms_list_for_summary = [Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17]
    if label or any(sym for sym in symptoms_list_for_summary if sym):
        with st.expander("📋 Current Input Summary"):
            st.write(f"**Disease:** {label}")
            symptoms_entered = []
            for i, sym in enumerate(symptoms_list_for_summary, 1):
                if sym and sym != '🖊️ Enter custom symptom':
                    symptoms_entered.append(f"Symptom {i}: {sym}")
            
            if symptoms_entered:
                st.write("**Symptoms:**")
                for symptom in symptoms_entered:
                    st.write(f"- {symptom}")
            else:
                st.write("No symptoms entered yet")

    if st.button("Submit"):
        if not label.strip():
            st.error("Please enter a disease label")
        elif not any(sym for sym in [Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17] if sym and sym != '🖊️ Enter custom symptom'):
            st.error("Please enter at least one symptom")
        else:
            if add_data(label, Sym_1, Sym_2, Sym_3, Sym_4, Sym_5, Sym_6, Sym_7, Sym_8, Sym_9, Sym_10, Sym_11, Sym_12, Sym_13, Sym_14, Sym_15, Sym_16, Sym_17):
                st.success('✅ Data insertion completed successfully. Thank you!')
                st.balloons()

# Train Model page
elif selected_option == 'Train Model':
    st.title('Model Training Page')
    st.header("Train the Model")
    st.write("Click the button to start training the model")

    def training_model():
        dataset_path = os.path.join(script_dir, 'Dataset', 'dataset.csv')
        if not os.path.exists(dataset_path):
            st.error(f"Dataset file not found at: {dataset_path}")
            return None
        if not os.path.exists(severity_path):
            st.error(f"Symptom severity file not found at: {severity_path}")
            return None

        try:
            dataset = pd.read_csv(dataset_path)
            for col in dataset.columns:
                if dataset[col].dtype == 'object':
                    dataset[col] = dataset[col].astype(str).str.replace('_', ' ').str.strip()

            dataset.fillna(0, inplace=True)
            severity = pd.read_csv(severity_path)
            severity['Symptom'] = severity['Symptom'].str.replace('_', ' ').str.strip()
            
            vals = dataset.values
            symp = severity['Symptom'].unique()
            cols = dataset.columns
            for i in range(len(symp)):
                vals[vals == symp[i]] = severity[severity['Symptom'] == symp[i]]['weight'].values[0]

            df = pd.DataFrame(vals, columns=cols)
            df = df.replace(['spotting  urination', 'dischromic  patches', 'foul smell of urine'], 0)
            
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            X = df.iloc[:, 1:].values
            y = df['Disease'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_RFC.fit(X_train, y_train)
            jb.dump(model_RFC, model_path)
            
            pred = model_RFC.predict(X_test)
            Acc = accuracy_score(y_test, pred)
            return Acc
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            return None

    # Show dataset info
    dataset_path = os.path.join(script_dir, 'Dataset', 'dataset.csv')
    if os.path.exists(dataset_path):
        try:
            dataset = pd.read_csv(dataset_path)
            st.info(f"📊 Dataset contains {len(dataset)} records")
            
            # Show disease distribution
            if 'Disease' in dataset.columns:
                disease_counts = dataset['Disease'].value_counts()
                st.write("**Disease Distribution:**")
                for disease, count in disease_counts.head(10).items():
                    st.write(f"- {disease}: {count} cases")
                if len(disease_counts) > 10:
                    st.write(f"... and {len(disease_counts) - 10} more diseases")
        except Exception as e:
            st.warning(f"Could not read dataset info: {str(e)}")

    if st.button("Start Training"):
        with st.spinner("Training model..."):
            Acc = training_model()
        if Acc is not None:
            st.success(f"✅ Model trained successfully with accuracy: {Acc*100:.2f}%")
            st.balloons()
            # Reload the page to update model status
            st.experimental_rerun()
        else:
            st.error("❌ Model training failed. Please check your data files.")