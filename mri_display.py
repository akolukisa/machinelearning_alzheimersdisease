import os
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from colorama import init, Fore, Style
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize colorama
init()

# Initialize global variables
accuracies = []
best_params_rf = {}
best_params_svm = {}
best_params_logreg = {}
best_params_dtree = {}


# Define the function determine_label before using it
def determine_label(group):
    if group == 'Demented' or group == 'Converted':
        return 1
    elif group == 'Nondemented':
        return 0
    else:
        return None

# Main directory where MRI files are located
data_dir = '/Users/akolukisa/mri1'

# Path to the text file containing demographic data
demographic_file = '/Users/akolukisa/demografik.txt'  # Make sure the file path is correct

# Read the demographic data text file
demographic_data = pd.read_csv(demographic_file, delimiter='\t')

# Add the Label column to demographic_data
demographic_data['Label'] = demographic_data['Group'].apply(determine_label)

# Create the image_labels dictionary
image_labels = {row['Subject ID']: row['Label'] for index, row in demographic_data.iterrows()}

patients_without_features = []
patients_without_labels = []

# Function to load NIfTI files
def load_nifti_file(img_filepath):
    try:
        nifti_img = nib.load(img_filepath)
        return nifti_img.get_fdata()
    except Exception as e:
        print(f"{Fore.RED}Error loading file {os.path.basename(img_filepath)}: {e}{Style.RESET_ALL}")
        return None

# Function to display the selected MRI
def display_selected_mri():
    selected_patient_id = combobox.get()
    if selected_patient_id:
        # Load and display MRI images for the selected patient
        mri_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.img') and selected_patient_id in root:
                    mri_files.append(os.path.join(root, file))

        if mri_files:
            for mri_file in mri_files:
                image_data = load_nifti_file(mri_file)
                if image_data is not None:
                    plt.figure()
                    plt.imshow(np.rot90(image_data[:, :, image_data.shape[2] // 2]), cmap='gray')
                    plt.title(f"{selected_patient_id} MRI")
                    plt.show()
                else:
                    print(f"{Fore.RED}Error loading MRI file: {mri_file}{Style.RESET_ALL}")
        else:
            messagebox.showerror("Error", "No MRI images found for the selected patient.")
    else:
        messagebox.showerror("Error", "Please select a patient.")

# Function to update combobox values
def update_combobox_values():
    patient_ids = set()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.img'):
                # Extract patient ID from the file path
                parts = root.split(os.sep)
                for part in parts:
                    if part.startswith('OAS2'):
                        patient_ids.add(part)
                        break

    combobox['values'] = list(patient_ids)
    if patient_ids:
        combobox.set(next(iter(patient_ids)))
    else:
        combobox.set("")

# Function to extract features from MRI data
def extract_features(image_data):
    return extract_grey_matter_density(image_data)

# Function to extract grey matter density
def extract_grey_matter_density(image_data):
    if image_data is None:
        return None
    grey_matter_density = np.mean(image_data)
    return grey_matter_density

def select_patient():
    global selected_patient_id
    selected_patient_id = combobox.get()
    if selected_patient_id:
        messagebox.showinfo("Hasta Seçildi", f"{selected_patient_id} hasta seçildi.")
    else:
        messagebox.showerror("Hata", "Lütfen bir hasta seçin.")


# Function to load MRI data and extract features for all patients
def step1():
    global features, labels, patient_ids, previous_patient_id
    features = []
    labels = []  # List to hold labels
    patient_ids = []

    patient_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    previous_patient_id = None

    for dir_name in patient_dirs:
        parts = dir_name.split('_')
        if len(parts) > 1:
            patient_id = parts[1]
            if patient_id != previous_patient_id:
                if previous_patient_id is not None:
                    print(f"{Fore.RED}{previous_patient_id}. Loading and scanning different MR images for the patient.{Style.RESET_ALL}")
                previous_patient_id = patient_id
            print(f"{Fore.GREEN}{patient_id}. Loading and scanning patient sample files.{Style.RESET_ALL}")
            patient_features = []
            loaded_files = []
            dir_path = os.path.join(data_dir, dir_name)
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.img'):
                        file_path = os.path.join(root, file)
                        image_data = load_nifti_file(file_path)
                        if image_data is not None:
                            loaded_files.append(os.path.basename(file_path))
                            feature = extract_features(image_data)
                            if feature is not None:
                                patient_features.append(feature)
            if loaded_files:
                print(f"{Fore.BLUE}Scanning files and extracting features: {', '.join(loaded_files)}{Style.RESET_ALL}")
            full_patient_id = f'OAS2_{patient_id}'
            if not patient_features:
                patients_without_features.append(full_patient_id)
                print(f"{Fore.YELLOW}Warning: No features extracted for {full_patient_id}!{Style.RESET_ALL}")
                continue

            label = image_labels.get(full_patient_id, None)
            if label is not None:
                avg_feature = np.mean(patient_features)
                features.append(avg_feature)
                labels.append(label)  # Add labels to the list
                patient_ids.append(full_patient_id)
                print(f"{Fore.GREEN}Adding label: {full_patient_id} - {label}{Style.RESET_ALL}")
            else:
                patients_without_labels.append(full_patient_id)
                print(f"{Fore.YELLOW}Warning: No label found for {full_patient_id}!{Style.RESET_ALL}")

    if previous_patient_id is not None:
        print(f"{Fore.RED}{previous_patient_id}. Loading and scanning different MR images for the patient.{Style.RESET_ALL}")

    print(f"{Fore.CYAN}Total number of processed patients: {len(patient_ids)}")
    print(f"Processed patient IDs: {patient_ids}{Style.RESET_ALL}")

    check_missing_data()

    # Update combobox values
    update_combobox_values()

# Function to check for missing data
def check_missing_data():
    print(f"{Fore.RED}Patients with missing features or labels:{Style.RESET_ALL}")
    if patients_without_features:
        print(f"{Fore.YELLOW}Patients with missing features:{Style.RESET_ALL}")
        for patient in patients_without_features:
            print(patient)
    if patients_without_labels:
        print(f"{Fore.YELLOW}Patients with missing labels:{Style.RESET_ALL}")
        for patient in patients_without_labels:
            print(patient)

X = None
y = None

# Function to combine features and labels with demographic data
# Function to combine features and labels with demographic data
# Function to combine features and labels with demographic data
def step2():
    global combined_data, X, y
    print(f"{Fore.YELLOW}Step-2: Converting features and labels to numpy arrays and merging with demographic data.{Style.RESET_ALL}")

    # Convert MRI features and labels to numpy arrays
    X_mri = np.array(features).reshape(-1, 1)
    y = np.array(labels)

    # Combine MRI and demographic data
    mri_data = pd.DataFrame({"Subject ID": patient_ids, "Grey Matter Density": X_mri.flatten(), "Label": y})

    # Print mri_data for debugging
    print(f"{Fore.GREEN}MRI Data:\n{mri_data.head()}{Style.RESET_ALL}")

    # Read demographic data
    demographic_data = pd.read_csv(demographic_file, delimiter='\t')

    # Print demographic_data for debugging
    print(f"{Fore.GREEN}Demographic Data before adding Label:\n{demographic_data.head()}{Style.RESET_ALL}")

    # Add the Label column to demographic_data
    demographic_data['Label'] = demographic_data['Group'].apply(determine_label)

    # Print demographic_data for debugging
    print(f"{Fore.GREEN}Demographic Data after adding Label:\n{demographic_data.head()}{Style.RESET_ALL}")

    # Merge MRI data with demographic data
    combined_data = pd.merge(mri_data, demographic_data, on='Subject ID', how='inner')

    # Print combined_data for debugging
    print(f"{Fore.GREEN}Combined Data after merging:\n{combined_data.head()}{Style.RESET_ALL}")

    # Ensure 'Label' column is present
    if 'Label_y' in combined_data.columns:
        combined_data.dropna(subset=['Label_y'], inplace=True)
        combined_data.rename(columns={'Label_y': 'Label'}, inplace=True)  # Rename Label_y to Label
    else:
        print(f"{Fore.RED}Error: 'Label' column not found in combined_data!{Style.RESET_ALL}")
        return

    # Encode categorical data
    label_encoders = {}
    for column in ['M/F', 'Hand']:
        le = LabelEncoder()
        combined_data[column] = le.fit_transform(combined_data[column])
        label_encoders[column] = le

    # Remove 'MRI ID', 'Group', 'Label_x' columns and keep only numerical data
    X = combined_data.drop(columns=['Subject ID', 'MRI ID', 'Group', 'Label_x', 'Label'])  # Do not drop 'Label' column
    y = combined_data['Label']

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Display data frame
    print(f"{Fore.CYAN}{combined_data.head()}{Style.RESET_ALL}")

    # Check patient data summary
    check_patient_data()




# Function to split data into training and test sets
# Function to split data into training and test sets
# Function to split data into training and test sets
# Function to split data into training and test sets
def step3():
    global X_train, X_test, y_train, y_test, train_df, test_df

    print(f"{Fore.YELLOW}Step-3: Splitting data into training and test sets with GroupKFold to prevent data leakage.{Style.RESET_ALL}")

    # Extract patient IDs for grouping
    groups = combined_data['Subject ID']

    # 'Group' kolonunu hedef değişken olarak belirle
    y = combined_data['Group'].apply(determine_label)

    # Özellik kolonlarını belirle (Group haricindeki tüm kolonlar)
    X = combined_data.drop(columns=['Subject ID', 'MRI ID', 'Group', 'Label'])

    # Create GroupKFold instance
    group_kfold = GroupKFold(n_splits=5)
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break



    # Display training and test sets
    train_df = X_train.copy()
    train_df['Group'] = y_train
    test_df = X_test.copy()
    test_df['Group'] = y_test

    # Check which patients are in training and test sets
    train_patient_ids = groups.iloc[train_index].unique()
    test_patient_ids = groups.iloc[test_index].unique()
    print(f"Patients in the training set: {train_patient_ids}")
    print(f"Patients in the test set: {test_patient_ids}")

    # Remove rows with missing labels
    train_df.dropna(subset=['Group'], inplace=True)
    test_df.dropna(subset=['Group'], inplace=True)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    train_df.iloc[:, :-1] = imputer.fit_transform(train_df.iloc[:, :-1])
    test_df.iloc[:, :-1] = imputer.transform(test_df.iloc[:, :-1])

    # Check and report missing values
    if train_df.isnull().values.any():
        print(f"{Fore.RED}There are missing values in the training set!{Style.RESET_ALL}")
    if test_df.isnull().values.any():
        print(f"{Fore.RED}There are missing values in the test set!{Style.RESET_ALL}")

    print(f"{Fore.CYAN}Training Set:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{train_df.head()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Test Set:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{test_df.head()}{Style.RESET_ALL}")

    messagebox.showinfo("Step 3", "Data has been split into training and test sets.")


# Replace the original step3 function with this new one and rerun the script to check if the issue is resolved.
def check_patient_data():
    print(f"{Fore.YELLOW}Checking data summary...{Style.RESET_ALL}")

    # Check the number of MRI scans each patient has
    patient_counts = combined_data['Subject ID'].value_counts()
    print(f"{Fore.CYAN}Number of MRI scans per patient:{Style.RESET_ALL}")
    print(patient_counts)

    # Check the number of unique patients
    unique_patients = combined_data['Subject ID'].nunique()
    total_rows = combined_data.shape[0]
    print(f"{Fore.CYAN}Total number of patients: {unique_patients}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Total number of rows in the data frame: {total_rows}{Style.RESET_ALL}")

    # Calculate mean for numeric columns only
    numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
    patient_summary = combined_data.groupby('Subject ID')[numeric_columns].mean().reset_index()

    # Add non-numeric columns back
    first_entries = combined_data.groupby('Subject ID').first().reset_index()
    non_numeric_columns = combined_data.columns.difference(numeric_columns)
    patient_summary = pd.merge(patient_summary, first_entries[non_numeric_columns], on='Subject ID', how='left')

    print(f"{Fore.CYAN}Summary representing each patient with a single row:{Style.RESET_ALL}")
    print(patient_summary.head())

# Function to plot ROC curve
# Function to plot ROC curve
# Function to evaluate model performance
def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}Accuracy: {accuracy}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Precision: {precision}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Recall: {recall}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}F1 Score: {f1}{Style.RESET_ALL}")

    print(f"{Fore.MAGENTA}Classification Report:\n{classification_report(y_test, y_pred)}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}{Style.RESET_ALL}")

    # Use predict_proba or decision_function based on the model
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        raise AttributeError("Model does not have predict_proba or decision_function method")

    # Modelin tahmin ettiği olasılık değerlerini yazdır
    print(f"{Fore.CYAN}Predicted probabilities: {y_proba}{Style.RESET_ALL}")

    plot_roc_curve(y_test, y_proba, model.__class__.__name__)

    return accuracy, precision, recall, f1

# Function to plot ROC curve
def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for ' + model_name)
    plt.legend(loc="lower right")
    plt.show()




def step3_stratified():
    global X_train, X_test, y_train, y_test, train_df, test_df

    print(f"{Fore.YELLOW}Step-3: Splitting data into stratified training and test sets.{Style.RESET_ALL}")

    # Stratified split
    stratified_split = StratifiedKFold(n_splits=5)
    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break  # Use only the first split

    # Extract correct feature column names
    column_names = combined_data.drop(columns=['Subject ID', 'MRI ID', 'Group', 'Label_x', 'Label']).columns.tolist()

    # Display training and test sets
    train_df = pd.DataFrame(X_train, columns=column_names)
    train_df['Label'] = y_train
    test_df = pd.DataFrame(X_test, columns=column_names)
    test_df['Label'] = y_test

    # Check which patients are in training and test sets
    train_patient_ids = combined_data.iloc[train_index]['Subject ID'].unique()
    test_patient_ids = combined_data.iloc[test_index]['Subject ID'].unique()
    print(f"Patients in the training set: {train_patient_ids}")
    print(f"Patients in the test set: {test_patient_ids}")

    # Remove rows with missing labels
    train_df.dropna(subset=['Label'], inplace=True)
    test_df.dropna(subset=['Label'], inplace=True)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    train_df.iloc[:, :-1] = imputer.fit_transform(train_df.iloc[:, :-1])
    test_df.iloc[:, :-1] = imputer.transform(test_df.iloc[:, :-1])

    # Check and report missing values
    if train_df.isnull().values.any():
        print(f"{Fore.RED}There are missing values in the training set!{Style.RESET_ALL}")
    if test_df.isnull().values.any():
        print(f"{Fore.RED}There are missing values in the test set!{Style.RESET_ALL}")

    print(f"{Fore.CYAN}Training Set:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{train_df.head()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Test Set:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{test_df.head()}{Style.RESET_ALL}")

    # Model eğitimi ve doğruluk hesaplaması
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    messagebox.showinfo("Step 3", "Data has been split into training and test sets.")



def step4_rf_optimized():
    global model, accuracies  # Ensure accuracies is recognized as a global variable
    print(f"{Fore.YELLOW}Step-4: Training optimized Random Forest model.{Style.RESET_ALL}")

    n_estimators = int(rf_estimators_entry.get())
    max_depth = int(rf_max_depth_entry.get())

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Step-5: Evaluating the model and calculating performance metrics.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate_model_performance(model, X_test, y_test)

    accuracies.append(('Random Forest', n_estimators, max_depth, accuracy, precision, recall, f1))

    messagebox.showinfo("Step 4", f"Optimized Random Forest model trained.\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


# Function to train and evaluate SVM model with optimization
def step4_svm_optimized():
    global model, accuracies
    print(f"{Fore.YELLOW}Step-4: Training optimized SVM model.{Style.RESET_ALL}")

    c_value = float(svm_c_entry.get())
    kernel = svm_kernel_entry.get()

    model = SVC(C=c_value, kernel=kernel, probability=True)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Step-5: Evaluating the model and calculating performance metrics.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate_model_performance(model, X_test, y_test)

    accuracies.append(('SVM', c_value, kernel, accuracy, precision, recall, f1))

    messagebox.showinfo("Step 4", f"Optimized SVM model trained.\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


# Function to train and evaluate Logistic Regression model with optimization
# Function to train and evaluate Logistic Regression model with optimization
def step4_logreg_optimized():
    global model, accuracies  # Ensure accuracies is recognized as a global variable
    print(f"{Fore.YELLOW}Step-4: Training optimized Logistic Regression model.{Style.RESET_ALL}")

    max_iter = int(logreg_max_iter_entry.get())
    solver = logreg_solver_entry.get()

    model = LogisticRegression(max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Step-5: Evaluating the model and calculating performance metrics.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate_model_performance(model, X_test, y_test)

    accuracies.append(('Logistic Regression', max_iter, solver, accuracy, precision, recall, f1))

    messagebox.showinfo("Step 4", f"Optimized Logistic Regression model trained.\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")



# Function to train and evaluate Decision Tree model with optimization
def step4_dtree_optimized():
    global model, accuracies
    print(f"{Fore.YELLOW}Step-4: Training optimized Decision Tree model.{Style.RESET_ALL}")

    max_depth = int(dtree_max_depth_entry.get())
    min_samples_split = int(dtree_min_samples_split_entry.get())

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Step-5: Evaluating the model and calculating performance metrics.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate_model_performance(model, X_test, y_test)

    accuracies.append(('Decision Tree', max_depth, min_samples_split, accuracy, precision, recall, f1))

    messagebox.showinfo("Step 4", f"Optimized Decision Tree model trained.\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")


# Function to optimize Random Forest hyperparameters
def optimize_rf_hyperparameters():
    global best_params_rf
    print(f"{Fore.YELLOW}Optimizing Random Forest hyperparameters...{Style.RESET_ALL}")
    param_grid = {
        'n_estimators': np.linspace(10, 500, num=10, dtype=int).tolist(),
        'max_depth': np.linspace(1, 100, num=10, dtype=int).tolist()
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')  # 5-fold cross-validation
    grid_search.fit(X_train, y_train)

    best_params_rf = grid_search.best_params_  # Update best_params_rf
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}Best hyperparameters: {best_params_rf}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Best cross-validation score: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test accuracy: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("Random Forest Hyperparameter Optimization", f"Best hyperparameters: {best_params_rf}\nTest Accuracy: {test_score}")

    # Show accuracy scores as 3D surface plot
    results = pd.DataFrame(grid_search.cv_results_)
    param_grid_n_estimators = np.array(param_grid['n_estimators'])
    param_grid_max_depth = np.array(param_grid['max_depth'])
    mean_test_scores = results['mean_test_score'].values.reshape(len(param_grid_n_estimators), len(param_grid_max_depth))

    n_estimators, max_depth = np.meshgrid(param_grid_n_estimators, param_grid_max_depth)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(n_estimators, max_depth, mean_test_scores.T, cmap='viridis')

    ax.set_xlabel('Number of Estimators')
    ax.set_ylabel('Max Depth')
    ax.set_zlabel('Mean Test Score')
    fig.colorbar(surf)

    plt.title('Random Forest Hyperparameter Optimization')
    plt.show()

# Function to retrain Random Forest with best hyperparameters
def retrain_with_best_params_rf():
    global model, accuracies

    rf_estimators_entry.delete(0, tk.END)
    rf_estimators_entry.insert(0, best_params_rf['n_estimators'])
    rf_max_depth_entry.delete(0, tk.END)
    rf_max_depth_entry.insert(0, best_params_rf['max_depth'])

    n_estimators = best_params_rf['n_estimators']
    max_depth = best_params_rf['max_depth']
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('Random Forest (Retrained)', n_estimators, max_depth, accuracy))

    messagebox.showinfo("Model Retrained", f"Random Forest model retrained with suggested hyperparameters.\nAccuracy: {accuracy}")

# Function to optimize SVM hyperparameters
def optimize_svm_hyperparameters():
    global best_params_svm
    print(f"{Fore.YELLOW}Optimizing SVM hyperparameters...{Style.RESET_ALL}")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    model = SVC()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params_svm = grid_search.best_params_  # Update best_params_svm
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}Best hyperparameters: {best_params_svm}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Best cross-validation score: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test accuracy: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("SVM Hyperparameter Optimization", f"Best hyperparameters: {best_params_svm}\nTest Accuracy: {test_score}")

    # Show accuracy scores as 3D surface plot
    results = pd.DataFrame(grid_search.cv_results_)
    C_values = np.array(param_grid['C'])
    kernel_values = results['param_kernel'].unique()
    mean_test_scores = results.pivot_table(index='param_C', columns='param_kernel', values='mean_test_score').values

    C_grid, kernel_grid = np.meshgrid(C_values, np.arange(len(kernel_values)))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(C_grid, kernel_grid, mean_test_scores.T, cmap='viridis')

    ax.set_xlabel('C value')
    ax.set_ylabel('Kernel')
    ax.set_zlabel('Mean Test Score')
    ax.set_yticks(np.arange(len(kernel_values)))
    ax.set_yticklabels(kernel_values)
    fig.colorbar(surf)

    plt.title('SVM Hyperparameter Optimization')
    plt.show()

# Function to retrain SVM with best hyperparameters
def retrain_with_best_params_svm():
    global model, accuracies

    svm_c_entry.delete(0, tk.END)
    svm_c_entry.insert(0, best_params_svm['C'])
    svm_kernel_entry.delete(0, tk.END)
    svm_kernel_entry.insert(0, best_params_svm['kernel'])

    c_value = best_params_svm['C']
    kernel = best_params_svm['kernel']
    model = SVC(C=c_value, kernel=kernel, probability=True)
    model.fit(X_train, y_train)  # Retrain model
    y_pred = model.predict(X_test)  # Update predictions
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('SVM (Retrained)', c_value, kernel, accuracy))  # Add updated accuracy

    messagebox.showinfo("Model Retrained", f"SVM model retrained with suggested hyperparameters.\nAccuracy: {accuracy}")

# Function to optimize Logistic Regression hyperparameters
def optimize_logreg_hyperparameters():
    global best_params_logreg  # Define global variable
    print(f"{Fore.YELLOW}Optimizing Logistic Regression hyperparameters...{Style.RESET_ALL}")

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'max_iter': np.linspace(100, 10000, num=10, dtype=int).tolist(),
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    model = LogisticRegression()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    best_params_logreg = grid_search.best_params_  # Update best_params_logreg
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test_scaled)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}Best hyperparameters: {best_params_logreg}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Best cross-validation score: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test accuracy: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("Logistic Regression Hyperparameter Optimization", f"Best hyperparameters: {best_params_logreg}\nTest Accuracy: {test_score}")

    # Show accuracy scores as 3D surface plot
    results = pd.DataFrame(grid_search.cv_results_)
    max_iter_values = np.array(param_grid['max_iter'])
    solver_values = np.array(param_grid['solver'])
    mean_test_scores = results.pivot_table(index='param_max_iter', columns='param_solver', values='mean_test_score').T

    max_iter_grid, solver_grid = np.meshgrid(max_iter_values, np.arange(len(solver_values)))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(max_iter_grid, solver_grid, mean_test_scores, cmap='viridis')

    ax.set_xlabel('Max Iter')
    ax.set_ylabel('Solver')
    ax.set_zlabel('Mean Test Score')
    ax.set_yticks(np.arange(len(solver_values)))
    ax.set_yticklabels(solver_values)
    fig.colorbar(surf)

    plt.title('Logistic Regression Hyperparameter Optimization')
    plt.show()

# Function to retrain Logistic Regression with best hyperparameters
def retrain_with_best_params_logreg():
    global model, accuracies

    logreg_max_iter_entry.delete(0, tk.END)
    logreg_max_iter_entry.insert(0, best_params_logreg['max_iter'])
    logreg_solver_entry.delete(0, tk.END)
    logreg_solver_entry.insert(0, best_params_logreg['solver'])

    max_iter = best_params_logreg['max_iter']
    solver = best_params_logreg['solver']
    model = LogisticRegression(max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train)  # Retrain model
    y_pred = model.predict(X_test)  # Update predictions
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('Logistic Regression (Retrained)', max_iter, solver, accuracy))  # Add updated accuracy

    messagebox.showinfo("Model Retrained", f"Logistic Regression model retrained with suggested hyperparameters.\nAccuracy: {accuracy}")

# Function to optimize Decision Tree hyperparameters
def optimize_dtree_hyperparameters():
    global best_params_dtree  # Define global variable
    print(f"{Fore.YELLOW}Optimizing Decision Tree hyperparameters...{Style.RESET_ALL}")
    param_grid = {
        'max_depth': np.linspace(1, 100, num=10, dtype=int).tolist(),
        'min_samples_split': np.linspace(2, 20, num=10, dtype=int).tolist()
    }
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params_dtree = grid_search.best_params_  # Update best_params_dtree
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}Best hyperparameters: {best_params_dtree}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Best cross-validation score: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test accuracy: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("Decision Tree Hyperparameter Optimization", f"Best hyperparameters: {best_params_dtree}\nTest Accuracy: {test_score}")

    # Show accuracy scores as 3D surface plot
    results = pd.DataFrame(grid_search.cv_results_)
    max_depth_values = np.array(param_grid['max_depth'])
    min_samples_split_values = np.array(param_grid['min_samples_split'])
    mean_test_scores = results['mean_test_score'].values.reshape(len(max_depth_values), len(min_samples_split_values))

    max_depth_grid, min_samples_split_grid = np.meshgrid(max_depth_values, min_samples_split_values)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(max_depth_grid, min_samples_split_grid, mean_test_scores.T, cmap='viridis')

    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Min Samples Split')
    ax.set_zlabel('Mean Test Score')
    fig.colorbar(surf)

    plt.title('Decision Tree Hyperparameter Optimization')
    plt.show()

# Function to retrain Decision Tree with best hyperparameters
def retrain_with_best_params_dtree():
    global model, accuracies

    dtree_max_depth_entry.delete(0, tk.END)
    dtree_max_depth_entry.insert(0, best_params_dtree['max_depth'])
    dtree_min_samples_split_entry.delete(0, tk.END)
    dtree_min_samples_split_entry.insert(0, best_params_dtree['min_samples_split'])

    max_depth = best_params_dtree['max_depth']
    min_samples_split = best_params_dtree['min_samples_split']
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, y_train)  # Retrain model
    y_pred = model.predict(X_test)  # Update predictions
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('Decision Tree (Retrained)', max_depth, min_samples_split, accuracy))  # Add updated accuracy

    messagebox.showinfo("Model Retrained", f"Decision Tree model retrained with suggested hyperparameters.\nAccuracy: {accuracy}")

# Function to show plots
def show_plots():
    # Show model accuracies as a table
    accuracy_table = pd.DataFrame(accuracies, columns=['Model', 'Param1', 'Param2', 'Accuracy', 'Precision', 'Recall', 'F1'])
    print("\nModel Accuracies")
    print(accuracy_table)

    # Show accuracies as a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=accuracy_table)
    plt.title('Model Accuracies')
    plt.show()

    # Show tables in Tkinter interface
    def show_table(table, title):
        window = tk.Toplevel(root)
        window.title(title)
        frame = ttk.Frame(window)
        frame.pack(fill='both', expand=True)
        table_widget = ttk.Treeview(frame, columns=list(table.columns), show='headings')
        for col in table.columns:
            table_widget.heading(col, text=col)
            table_widget.column(col, width=100)
        for row in table.itertuples(index=False):
            table_widget.insert("", "end", values=row)
        table_widget.pack(fill='both', expand=True)
        scrollbar = ttk.Scrollbar(window, orient='vertical', command=table_widget.yview)
        table_widget.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

    show_table(accuracy_table, "Model Accuracies")

import tkinter as tk
from tkinter import ttk

import tkinter as tk
from tkinter import ttk

# Create GUI
root = tk.Tk()
root.title("MRI Analysis Steps")

# Step 1 and Step 2 buttons
step1_button = tk.Button(root, text="Step 1: Load MRI Files", command=step1, bg="#ff9999")
step1_button.pack(pady=(5, 2))  # Reduced padding

step2_button = tk.Button(root, text="Step 2: Convert Features and Labels", command=step2, bg="#ff9999")
step2_button.pack(pady=(2, 10))  # Reduced padding with space after this step

# Entry boxes for train and test set ratios
ratio_frame = tk.Frame(root)
ratio_frame.pack(pady=5)  # Reduced padding

tk.Label(ratio_frame, text="Training Set Ratio:").grid(row=0, column=0, padx=5)
train_size_entry = tk.Entry(ratio_frame, width=5)  # Reduced width
train_size_entry.insert(0, "0.8")  # Default value
train_size_entry.grid(row=0, column=1, padx=5)

tk.Label(ratio_frame, text="Test Set Ratio:").grid(row=0, column=2, padx=5)
test_size_entry = tk.Entry(ratio_frame, width=5)  # Reduced width
test_size_entry.insert(0, "0.2")  # Default value
test_size_entry.grid(row=0, column=3, padx=5)

# Step 3 button
step3_button = tk.Button(root, text="Step 3: Split Data into Training and Test Sets", command=step3_stratified, bg="#ffcc99")
step3_button.pack(pady=(10, 10))  # Reduced padding with space before this step

# Model frame
model_frame = tk.Frame(root)
model_frame.pack(pady=5)  # Reduced padding

# Entry boxes for Random Forest hyperparameters
tk.Label(model_frame, text="RF Estimators:").grid(row=0, column=0, padx=5)
rf_estimators_entry = tk.Entry(model_frame, width=5)  # Reduced width
rf_estimators_entry.insert(0, "100")  # Default value
rf_estimators_entry.grid(row=0, column=1, padx=5)

tk.Label(model_frame, text="RF Max Depth:").grid(row=0, column=2, padx=5)
rf_max_depth_entry = tk.Entry(model_frame, width=5)  # Reduced width
rf_max_depth_entry.insert(0, "10")  # Default value
rf_max_depth_entry.grid(row=0, column=3, padx=5)

# RF buttons
step4_rf_button = tk.Button(model_frame, text="RF Accuracy", command=step4_rf_optimized, bg="#ff9999")
step4_rf_button.grid(row=1, column=0, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_rf_optimize_button = tk.Button(model_frame, text="Optimize RF", command=optimize_rf_hyperparameters, bg="#ff9999")
step4_rf_optimize_button.grid(row=1, column=1, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_rf_retrain_button = tk.Button(model_frame, text="Retrain RF", command=retrain_with_best_params_rf, bg="#ff9999")
step4_rf_retrain_button.grid(row=1, column=2, padx=5, pady=(2, 10))  # Reduced padding with space after this step

# Entry boxes for SVM hyperparameters
tk.Label(model_frame, text="SVM C:").grid(row=2, column=0, padx=5)
svm_c_entry = tk.Entry(model_frame, width=5)  # Reduced width
svm_c_entry.insert(0, "1.0")  # Default value
svm_c_entry.grid(row=2, column=1, padx=5)

tk.Label(model_frame, text="SVM Kernel:").grid(row=2, column=2, padx=5)
svm_kernel_entry = tk.Entry(model_frame, width=5)  # Reduced width
svm_kernel_entry.insert(0, "rbf")  # Default value
svm_kernel_entry.grid(row=2, column=3, padx=5)

# SVM buttons
step4_svm_button = tk.Button(model_frame, text="SVM Accuracy", command=step4_svm_optimized, bg="#99ccff")
step4_svm_button.grid(row=3, column=0, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_svm_optimize_button = tk.Button(model_frame, text="Optimize SVM", command=optimize_svm_hyperparameters, bg="#99ccff")
step4_svm_optimize_button.grid(row=3, column=1, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_svm_retrain_button = tk.Button(model_frame, text="Retrain SVM", command=retrain_with_best_params_svm, bg="#99ccff")
step4_svm_retrain_button.grid(row=3, column=2, padx=5, pady=(2, 10))  # Reduced padding with space after this step

# Entry boxes for Logistic Regression hyperparameters
tk.Label(model_frame, text="LogReg Iter:").grid(row=4, column=0, padx=5)
logreg_max_iter_entry = tk.Entry(model_frame, width=5)  # Reduced width
logreg_max_iter_entry.insert(0, "1000")  # Default value
logreg_max_iter_entry.grid(row=4, column=1, padx=5)

tk.Label(model_frame, text="LogReg Solver:").grid(row=4, column=2, padx=5)
logreg_solver_entry = tk.Entry(model_frame, width=5)  # Reduced width
logreg_solver_entry.insert(0, "lbfgs")  # Default value
logreg_solver_entry.grid(row=4, column=3, padx=5)

# LogReg buttons
step4_logreg_button = tk.Button(model_frame, text="LogReg Accuracy", command=step4_logreg_optimized, bg="#ffccff")
step4_logreg_button.grid(row=5, column=0, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_logreg_optimize_button = tk.Button(model_frame, text="Optimize LogReg", command=optimize_logreg_hyperparameters, bg="#ffccff")
step4_logreg_optimize_button.grid(row=5, column=1, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_logreg_retrain_button = tk.Button(model_frame, text="Retrain LogReg", command=retrain_with_best_params_logreg, bg="#ffccff")
step4_logreg_retrain_button.grid(row=5, column=2, padx=5, pady=(2, 10))  # Reduced padding with space after this step

# Entry boxes for Decision Tree hyperparameters
tk.Label(model_frame, text="DTree Depth:").grid(row=6, column=0, padx=5)
dtree_max_depth_entry = tk.Entry(model_frame, width=5)  # Reduced width
dtree_max_depth_entry.insert(0, "10")  # Default value
dtree_max_depth_entry.grid(row=6, column=1, padx=5)

tk.Label(model_frame, text="DTree Split:").grid(row=6, column=2, padx=5)
dtree_min_samples_split_entry = tk.Entry(model_frame, width=5)  # Reduced width
dtree_min_samples_split_entry.insert(0, "2")  # Default value
dtree_min_samples_split_entry.grid(row=6, column=3, padx=5)

# DTree buttons
step4_dtree_button = tk.Button(model_frame, text="DTree Accuracy", command=step4_dtree_optimized, bg="#ff9966")
step4_dtree_button.grid(row=7, column=0, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_dtree_optimize_button = tk.Button(model_frame, text="Optimize DTree", command=optimize_dtree_hyperparameters, bg="#ff9966")
step4_dtree_optimize_button.grid(row=7, column=1, padx=5, pady=(2, 10))  # Reduced padding with space after this step

step4_dtree_retrain_button = tk.Button(model_frame, text="Retrain DTree", command=retrain_with_best_params_dtree, bg="#ff9966")
step4_dtree_retrain_button.grid(row=7, column=2, padx=5, pady=(2, 10))  # Reduced padding with space after this step

# Show plots button
show_plots_button = tk.Button(root, text="Show Plots", command=show_plots, bg="#cccccc")
show_plots_button.pack(pady=(10, 5))  # Reduced padding

# MRI image display frame
selection_frame = tk.Frame(root)
selection_frame.pack(pady=5)  # Reduced padding

mri_label = tk.Label(selection_frame, text="Display MRI Images")  # Add title
mri_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))  # Reduced padding

combobox = ttk.Combobox(selection_frame)
combobox.grid(row=1, column=0, padx=5)  # Reduced padding

select_button = tk.Button(selection_frame, text="Select", command=select_patient, bg="#99ccff")
select_button.grid(row=1, column=1, padx=5, pady=2)  # Reduced padding

reflect_button = tk.Button(selection_frame, text="Show", command=display_selected_mri, bg="#99ccff")
reflect_button.grid(row=1, column=2, padx=5, pady=2)  # Reduced padding

root.mainloop()
