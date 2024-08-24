import os
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from colorama import init, Fore, Style
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Initialize colorama
init()

# MRI dosyalarının bulunduğu ana dizini belirtiliyor
data_dir = '/Users/akolukisa/mri3'

# Demografik verileri içeren txt dosyasının yolu
demographic_file = '/Users/akolukisa/demografik.txt'  # Dosya yolunu doğru belirttiğinizden emin olun

# Demografik verileri içeren txt dosyasını oku
demographic_data = pd.read_csv(demographic_file, delimiter='\t')  # Eğer veriler tab ile ayrılmışsa


# Alzheimer etiketlerini belirleme
def determine_label(group):
    if group == 'Demented' or group == 'Converted':
        return 1
    elif group == 'Nondemented':
        return 0
    else:
        return None


# Etiket sütununu ekle
demographic_data['Label'] = demographic_data['Group'].apply(determine_label)

# image_labels sözlüğünü oluştur
image_labels = {row['Subject ID']: row['Label'] for index, row in demographic_data.iterrows()}

patients_without_features = []
patients_without_labels = []


# MRI verilerini yükleme ve özellik çıkarma
def load_nifti_file(img_filepath):
    try:
        nifti_img = nib.load(img_filepath)
        return nifti_img.get_fdata()
    except Exception as e:
        print(f"{Fore.RED}Error loading file {os.path.basename(img_filepath)}: {e}{Style.RESET_ALL}")
        return None

def display_selected_mri():
    selected_patient_id = combobox.get()
    if selected_patient_id:
        # Seçilen hastanın MRI görüntülerini yükleme ve görüntüleme işlemi
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
            messagebox.showerror("Hata", "Seçilen hasta için MRI görüntüsü bulunamadı.")
    else:
        messagebox.showerror("Hata", "Lütfen bir hasta seçin.")


def update_combobox_values():
    patient_ids = set()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.img'):
                # Dosya yolundan hasta ID'sini ayıklama
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




def extract_features(image_data):
    return extract_grey_matter_density(image_data)


def extract_grey_matter_density(image_data):
    if image_data is None:
        return None
    grey_matter_density = np.mean(image_data)
    return grey_matter_density


# Tüm hastaları ve MRI verilerini yükleme
def step1():
    global features, labels, patient_ids, previous_patient_id
    features = []
    labels = []
    patient_ids = []

    patient_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    previous_patient_id = None

    for dir_name in patient_dirs:
        parts = dir_name.split('_')
        if len(parts) > 1:
            patient_id = parts[1]
            if patient_id != previous_patient_id:
                if previous_patient_id is not None:
                    print(
                        f"{Fore.RED}{previous_patient_id}. Hastaya ait farklı MR görüntüleri yükleniyor ve taranıyor.{Style.RESET_ALL}")
                previous_patient_id = patient_id
            print(f"{Fore.GREEN}{patient_id}. Hasta örneği dosyalar yükleniyor ve taranıyor.{Style.RESET_ALL}")
            patient_features = []
            loaded_files = []
            dir_path = os.path.join(data_dir, dir_name)
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isdir(file_path):
                    for subfile in os.listdir(file_path):
                        if subfile.endswith('.img'):
                            subfile_path = os.path.join(file_path, subfile)
                            image_data = load_nifti_file(subfile_path)
                            if image_data is not None:
                                loaded_files.append(os.path.basename(subfile_path))
                                feature = extract_features(image_data)
                                if feature is not None:
                                    patient_features.append(feature)
            if loaded_files:
                print(
                    f"{Fore.BLUE}Dosyalar taranıyor ve özellikler çıkarılıyor: {', '.join(loaded_files)}{Style.RESET_ALL}")
            full_patient_id = f'OAS2_{patient_id}'
            if not patient_features:
                patients_without_features.append(full_patient_id)
                print(f"{Fore.YELLOW}Uyarı: {full_patient_id} için özellik çıkarılamadı!{Style.RESET_ALL}")
                continue

            label = image_labels.get(full_patient_id, None)
            if label is not None:
                avg_feature = np.mean(patient_features)
                features.append(avg_feature)
                labels.append(label)
                patient_ids.append(full_patient_id)
                print(f"{Fore.GREEN}Etiket ekleniyor: {full_patient_id} - {label}{Style.RESET_ALL}")
            else:
                patients_without_labels.append(full_patient_id)
                print(f"{Fore.YELLOW}Uyarı: {full_patient_id} için etiket bulunamadı!{Style.RESET_ALL}")

    if previous_patient_id is not None:
        print(
            f"{Fore.RED}{previous_patient_id}. Hastaya ait farklı MR görüntüleri yükleniyor ve taranıyor.{Style.RESET_ALL}")

    print(f"{Fore.CYAN}Toplam işlenen hasta sayısı: {len(patient_ids)}")
    print(f"İşlenen hastaların kimlikleri: {patient_ids}{Style.RESET_ALL}")

    check_missing_data()

    # ComboBox'ı güncelleme
    update_combobox_values()


def check_missing_data():
    print(f"{Fore.RED}Özellik çıkarılamayan veya etiketi bulunmayan hastalar:{Style.RESET_ALL}")
    if patients_without_features:
        print(f"{Fore.YELLOW}Özellik çıkarılamayan hastalar:{Style.RESET_ALL}")
        for patient in patients_without_features:
            print(patient)
    if patients_without_labels:
        print(f"{Fore.YELLOW}Etiketi bulunmayan hastalar:{Style.RESET_ALL}")
        for patient in patients_without_labels:
            print(patient)


def step2():
    global combined_data, X, y
    print(f"{Fore.YELLOW}Adım-2: Özellikler ve etiketler numpy dizilerine dönüştürülüyor ve demografik verilerle birleştiriliyor.{Style.RESET_ALL}")

    # MRI verilerini ve etiketlerini numpy dizilerine dönüştürme
    X_mri = np.array(features).reshape(-1, 1)  # MRI özelliklerini uygun boyutta şekillendirme
    y = np.array(labels)

    # Uzunlukların eşleşip eşleşmediğini kontrol etme
    if len(patient_ids) != len(features) or len(features) != len(labels):
        print(f"{Fore.RED}Hata: patient_ids, features ve labels listelerinin uzunlukları eşleşmiyor!{Style.RESET_ALL}")
        return

    # Demografik verileri yükleme
    demographic_data = pd.read_csv(demographic_file, delimiter='\t')  # txt dosyasının ayraçlı olduğunu varsayalım

    # Demografik verileri kontrol et
    print(demographic_data.head())
    print(demographic_data.columns)

    # MRI ve demografik verileri birleştirme
    demographic_data = demographic_data[demographic_data['Subject ID'].isin(patient_ids)]
    combined_data = pd.DataFrame({"Subject ID": patient_ids, "Grey Matter Density": X_mri.flatten(), "Label": y})
    combined_data = pd.merge(combined_data, demographic_data, on='Subject ID')

    # NaN değerleri kontrol etme ve kaldırma
    combined_data.dropna(subset=['Label'], inplace=True)

    # Kategorik verileri kodlama
    label_encoders = {}
    for column in ['M/F', 'Hand', 'Group']:
        le = LabelEncoder()
        combined_data[column] = le.fit_transform(combined_data[column])
        label_encoders[column] = le

    # 'MRI ID' sütununu kaldırarak sadece sayısal verilerle işlem yapma
    X = combined_data.drop(columns=['Subject ID', 'Label', 'MRI ID'])
    y = combined_data['Label']

    # Eksik değerleri ortalama ile doldurma
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Verileri dataframe olarak göstermek
    print(f"{Fore.CYAN}{combined_data.head()}{Style.RESET_ALL}")

    # Veri özet kontrolü
    check_patient_data()



def step3():
    global X_train, X_test, y_train, y_test, train_df, test_df, combined_data

    try:
        test_size = float(test_size_entry.get())
        train_size = float(train_size_entry.get())
    except ValueError:
        messagebox.showerror("Hata", "Lütfen geçerli bir oran girin.")
        return

    if not (0 < train_size < 1 and 0 < test_size < 1 and train_size + test_size == 1):
        messagebox.showerror("Hata", "Lütfen geçerli bir eğitim ve test oranı girin. Oranların toplamı 1 olmalıdır.")
        return

    print(f"{Fore.YELLOW}Adım-3: Veriler eğitim ve test setlerine ayrılıyor.{Style.RESET_ALL}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size,
                                                        random_state=42)

    # Eğitim ve test setlerini dataframe olarak göstermek
    train_df = pd.DataFrame(X_train, columns=combined_data.drop(columns=['Subject ID', 'Label', 'MRI ID']).columns)
    train_df['Label'] = y_train
    test_df = pd.DataFrame(X_test, columns=combined_data.drop(columns=['Subject ID', 'Label', 'MRI ID']).columns)
    test_df['Label'] = y_test

    # Eğitim ve test setlerinde hangi hastaların olduğunu kontrol et
    train_patient_ids = combined_data.iloc[train_df.index]['Subject ID']
    test_patient_ids = combined_data.iloc[test_df.index]['Subject ID']
    print(f"Eğitim setindeki hastalar: {train_patient_ids.unique()}")
    print(f"Test setindeki hastalar: {test_patient_ids.unique()}")

    # Eksik etiketlere sahip satırları kaldırma
    train_df.dropna(subset=['Label'], inplace=True)
    test_df.dropna(subset=['Label'], inplace=True)

    # Eksik değerleri ortalama ile doldurma
    imputer = SimpleImputer(strategy='mean')
    train_df.iloc[:, :-1] = imputer.fit_transform(train_df.iloc[:, :-1])
    test_df.iloc[:, :-1] = imputer.transform(test_df.iloc[:, :-1])

    # NaN değerleri kontrol etme ve bildirme
    if train_df.isnull().values.any():
        print(f"{Fore.RED}Eğitim setinde eksik değerler var!{Style.RESET_ALL}")
    if test_df.isnull().values.any():
        print(f"{Fore.RED}Test setinde eksik değerler var!{Style.RESET_ALL}")

    print(f"{Fore.CYAN}Eğitim Seti:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{train_df.head()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Test Seti:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{test_df.head()}{Style.RESET_ALL}")

    messagebox.showinfo("Adım 3", "Veriler eğitim ve test setlerine ayrıldı.")


def check_patient_data():
    print(f"{Fore.YELLOW}Veri özet kontrolü yapılıyor...{Style.RESET_ALL}")

    # Her hastanın kaç adet MRI taramasına sahip olduğunu kontrol etme
    patient_counts = combined_data['Subject ID'].value_counts()
    print(f"{Fore.CYAN}Her hastanın kaç MRI taramasına sahip olduğu:{Style.RESET_ALL}")
    print(patient_counts)

    # Eşsiz hasta sayısını kontrol etme
    unique_patients = combined_data['Subject ID'].nunique()
    total_rows = combined_data.shape[0]
    print(f"{Fore.CYAN}Toplam hastaların sayısı: {unique_patients}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Veri çerçevesindeki toplam satır sayısı: {total_rows}{Style.RESET_ALL}")

    # Sadece sayısal sütunlar üzerinde ortalama hesaplama
    numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
    patient_summary = combined_data.groupby('Subject ID')[numeric_columns].mean().reset_index()

    # Sayısal olmayan sütunları geri ekleme
    first_entries = combined_data.groupby('Subject ID').first().reset_index()
    non_numeric_columns = combined_data.columns.difference(numeric_columns)
    patient_summary = pd.merge(patient_summary, first_entries[non_numeric_columns], on='Subject ID', how='left')

    print(f"{Fore.CYAN}Her hastanın sadece bir satırla temsil edildiği özet:{Style.RESET_ALL}")
    print(patient_summary.head())


# Global accuracies değişkenini tanımlama
accuracies = []


def step4_rf_optimized():
    global model, accuracies
    print(f"{Fore.YELLOW}Adım-4: Optimizasyonlu Random Forest modeli eğitiliyor.{Style.RESET_ALL}")

    n_estimators = int(rf_estimators_entry.get())
    max_depth = int(rf_max_depth_entry.get())

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Adım-5: Model değerlendirilip doğruluk oranı hesaplanıyor.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('Random Forest', n_estimators, max_depth, accuracy))
    print(f"{Fore.MAGENTA}Accuracy: {accuracy}{Style.RESET_ALL}")

    messagebox.showinfo("Adım 4", f"Optimizasyonlu Random Forest modeli eğitildi. Doğruluk: {accuracy}")


def step4_svm_optimized():
    global model, accuracies
    print(f"{Fore.YELLOW}Adım-4: Optimizasyonlu SVM modeli eğitiliyor.{Style.RESET_ALL}")

    c_value = float(svm_c_entry.get())
    kernel = svm_kernel_entry.get()

    model = SVC(C=c_value, kernel=kernel)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Adım-5: Model değerlendirilip doğruluk oranı hesaplanıyor.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('SVM', c_value, kernel, accuracy))
    print(f"{Fore.MAGENTA}Accuracy: {accuracy}{Style.RESET_ALL}")

    messagebox.showinfo("Adım 4", f"Optimizasyonlu SVM modeli eğitildi. Doğruluk: {accuracy}")


def step4_logreg_optimized():
    global model, accuracies
    print(f"{Fore.YELLOW}Adım-4: Optimizasyonlu Lojistik Regresyon modeli eğitiliyor.{Style.RESET_ALL}")

    max_iter = int(logreg_max_iter_entry.get())
    solver = logreg_solver_entry.get()

    model = LogisticRegression(max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Adım-5: Model değerlendirilip doğruluk oranı hesaplanıyor.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('Logistic Regression', max_iter, solver, accuracy))
    print(f"{Fore.MAGENTA}Accuracy: {accuracy}{Style.RESET_ALL}")

    messagebox.showinfo("Adım 4", f"Optimizasyonlu Lojistik Regresyon modeli eğitildi. Doğruluk: {accuracy}")


def step4_dtree_optimized():
    global model, accuracies
    print(f"{Fore.YELLOW}Adım-4: Optimizasyonlu Karar Ağacı modeli eğitiliyor.{Style.RESET_ALL}")

    max_depth = int(dtree_max_depth_entry.get())
    min_samples_split = int(dtree_min_samples_split_entry.get())

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, y_train)

    print(f"{Fore.YELLOW}Adım-5: Model değerlendirilip doğruluk oranı hesaplanıyor.{Style.RESET_ALL}")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(('Decision Tree', max_depth, min_samples_split, accuracy))
    print(f"{Fore.MAGENTA}Accuracy: {accuracy}{Style.RESET_ALL}")

    messagebox.showinfo("Adım 4", f"Optimizasyonlu Karar Ağacı modeli eğitildi. Doğruluk: {accuracy}")


def optimize_rf_hyperparameters():
    print(f"{Fore.YELLOW}Random Forest hiperparametre optimizasyonu yapılıyor...{Style.RESET_ALL}")
    param_grid = {
        'n_estimators': np.linspace(10, 500, num=10, dtype=int).tolist(),
        'max_depth': np.linspace(1, 100, num=10, dtype=int).tolist()
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}En iyi hiperparametreler: {best_params}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}En iyi çapraz doğrulama skoru: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test doğruluğu: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("Random Forest Hiperparametre Optimizasyonu", f"En iyi hiperparametreler: {best_params}\nTest Doğruluğu: {test_score}")

    # Hiperparametre değerlerine göre doğruluk oranlarını 3D surface plot ile gösterme
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


def optimize_svm_hyperparameters():
    print(f"{Fore.YELLOW}SVM hiperparametre optimizasyonu yapılıyor...{Style.RESET_ALL}")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    model = SVC()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}En iyi hiperparametreler: {best_params}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}En iyi çapraz doğrulama skoru: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test doğruluğu: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("SVM Hiperparametre Optimizasyonu", f"En iyi hiperparametreler: {best_params}\nTest Doğruluğu: {test_score}")

    # Hiperparametre değerlerine göre doğruluk oranlarını 3D surface plot ile gösterme
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


def optimize_logreg_hyperparameters():
    print(f"{Fore.YELLOW}Lojistik Regresyon hiperparametre optimizasyonu yapılıyor...{Style.RESET_ALL}")

    # Verileri ölçeklendirme
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

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test_scaled)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}En iyi hiperparametreler: {best_params}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}En iyi çapraz doğrulama skoru: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test doğruluğu: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("Lojistik Regresyon Hiperparametre Optimizasyonu", f"En iyi hiperparametreler: {best_params}\nTest Doğruluğu: {test_score}")

    # Hiperparametre değerlerine göre doğruluk oranlarını 3D surface plot ile gösterme
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


def optimize_dtree_hyperparameters():
    print(f"{Fore.YELLOW}Karar Ağacı hiperparametre optimizasyonu yapılıyor...{Style.RESET_ALL}")
    param_grid = {
        'max_depth': np.linspace(1, 100, num=10, dtype=int).tolist(),
        'min_samples_split': np.linspace(2, 20, num=10, dtype=int).tolist()
    }
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)

    print(f"{Fore.MAGENTA}En iyi hiperparametreler: {best_params}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}En iyi çapraz doğrulama skoru: {best_score}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Test doğruluğu: {test_score}{Style.RESET_ALL}")

    messagebox.showinfo("Karar Ağacı Hiperparametre Optimizasyonu", f"En iyi hiperparametreler: {best_params}\nTest Doğruluğu: {test_score}")

    # Hiperparametre değerlerine göre doğruluk oranlarını 3D surface plot ile gösterme
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


def update_combobox_values():
    combobox['values'] = patient_ids


def display_selected_mri():
    selected_patient_id = combobox.get()
    if selected_patient_id:
        # Seçilen hastanın MRI görüntülerini yükleme ve görüntüleme işlemi
        mri_files = []
        for d in os.listdir(data_dir):
            parts = d.split('_')
            if len(parts) > 1 and parts[1] == selected_patient_id:
                dir_path = os.path.join(data_dir, d)
                for f in os.listdir(dir_path):
                    if f.endswith('.img'):
                        mri_files.append(os.path.join(dir_path, f))

        if mri_files:
            for mri_file in mri_files:
                image_data = load_nifti_file(mri_file)
                if image_data is not None:
                    plt.figure()
                    plt.imshow(np.rot90(image_data[:, :, image_data.shape[2] // 2]), cmap='gray')
                    plt.title(f"{selected_patient_id} MRI")
                    plt.show()
        else:
            messagebox.showerror("Hata", "Seçilen hasta için MRI görüntüsü bulunamadı.")
    else:
        messagebox.showerror("Hata", "Lütfen bir hasta seçin.")


def show_plots():
    # Model Doğruluk Oranlarını Tablo Olarak Gösterme
    accuracy_table = pd.DataFrame(accuracies, columns=['Model', 'Param1', 'Param2', 'Accuracy'])
    print("\nModel Doğruluk Oranları")
    print(accuracy_table)

    # Doğruluk oranlarını grafikle gösterme
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', hue='Param1', data=accuracy_table)
    plt.title('Model Doğruluk Oranları')
    plt.show()

    # Tabloları Tkinter arayüzünde gösterme
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

    show_table(accuracy_table, "Model Doğruluk Oranları")


# Arayüz oluşturma
root = tk.Tk()
root.title("MRI Analiz Adımları")

step1_button = tk.Button(root, text="Adım 1: MRI Dosyalarını Yükle", command=step1)
step1_button.pack(pady=10)

step2_button = tk.Button(root, text="Adım 2: Özellikleri ve Etiketleri Dönüştür", command=step2)
step2_button.pack(pady=10)

# Eğitim ve test seti oranlarını girmek için giriş kutuları
ratio_frame = tk.Frame(root)
ratio_frame.pack(pady=10)

tk.Label(ratio_frame, text="Eğitim Seti Oranı:").grid(row=0, column=0)
train_size_entry = tk.Entry(ratio_frame, width=7)
train_size_entry.insert(0, "0.8")  # Varsayılan değer
train_size_entry.grid(row=0, column=1, padx=10)

tk.Label(ratio_frame, text="Test Seti Oranı:").grid(row=0, column=2)
test_size_entry = tk.Entry(ratio_frame, width=7)
test_size_entry.insert(0, "0.2")  # Varsayılan değer
test_size_entry.grid(row=0, column=3, padx=10)

step3_button = tk.Button(root, text="Adım 3: Verileri Eğitim ve Test Setlerine Ayır", command=step3)
step3_button.pack(pady=10)

model_frame = tk.Frame(root)
model_frame.pack(pady=10)

# Random Forest hiperparametre girişleri
tk.Label(model_frame, text="RF Estimators (Range: 10-500):").grid(row=0, column=0)
rf_estimators_entry = tk.Entry(model_frame, width=7)
rf_estimators_entry.insert(0, "100")  # Varsayılan değer
rf_estimators_entry.grid(row=0, column=1, padx=10)

tk.Label(model_frame, text="RF Max Depth (Range: 1-100):").grid(row=0, column=2)
rf_max_depth_entry = tk.Entry(model_frame, width=7)
rf_max_depth_entry.insert(0, "10")  # Varsayılan değer
rf_max_depth_entry.grid(row=0, column=3, padx=10)

step4_rf_button = tk.Button(model_frame, text="RF Modeli - Accuracy Göster", command=step4_rf_optimized)
step4_rf_button.grid(row=1, column=0, padx=5)

# Yeni eklenen buton
step4_rf_optimize_button = tk.Button(model_frame, text="RF Hiperparametre Optimize Et", command=optimize_rf_hyperparameters)
step4_rf_optimize_button.grid(row=1, column=1, padx=5)

# SVM hiperparametre girişleri
tk.Label(model_frame, text="SVM C (Range: 0.1-10.0):").grid(row=3, column=0)
svm_c_entry = tk.Entry(model_frame, width=7)
svm_c_entry.insert(0, "1.0")  # Varsayılan değer
svm_c_entry.grid(row=3, column=1, padx=10)

tk.Label(model_frame, text="SVM Kernel (linear/poly/rbf/sigmoid):").grid(row=3, column=2)
svm_kernel_entry = tk.Entry(model_frame, width=7)
svm_kernel_entry.insert(0, "rbf")  # Varsayılan değer
svm_kernel_entry.grid(row=3, column=3, padx=10)

step4_svm_button = tk.Button(model_frame, text="SVM Modeli - Accuracy Göster", command=step4_svm_optimized)
step4_svm_button.grid(row=4, column=0, padx=5)

# Yeni eklenen buton
step4_svm_optimize_button = tk.Button(model_frame, text="SVM Hiperparametre Optimize Et", command=optimize_svm_hyperparameters)
step4_svm_optimize_button.grid(row=4, column=1, padx=5)

# Logistic Regression hiperparametre girişleri
tk.Label(model_frame, text="LogReg Max Iter (Range: 100-10000):").grid(row=6, column=0)
logreg_max_iter_entry = tk.Entry(model_frame, width=7)
logreg_max_iter_entry.insert(0, "1000")  # Varsayılan değer
logreg_max_iter_entry.grid(row=6, column=1, padx=10)

tk.Label(model_frame, text="LogReg Solver (newton-cg/lbfgs/liblinear/sag/saga):").grid(row=6, column=2)
logreg_solver_entry = tk.Entry(model_frame, width=7)
logreg_solver_entry.insert(0, "lbfgs")  # Varsayılan değer
logreg_solver_entry.grid(row=6, column=3, padx=10)

step4_logreg_button = tk.Button(model_frame, text="LogReg Modeli - Accuracy Göster", command=step4_logreg_optimized)
step4_logreg_button.grid(row=7, column=0, padx=5)

# Yeni eklenen buton
step4_logreg_optimize_button = tk.Button(model_frame, text="LogReg Hiperparametre Optimize Et", command=optimize_logreg_hyperparameters)
step4_logreg_optimize_button.grid(row=7, column=1, padx=5)

# Decision Tree hiperparametre girişleri
tk.Label(model_frame, text="DTree Max Depth (Range: 1-100):").grid(row=9, column=0)
dtree_max_depth_entry = tk.Entry(model_frame, width=7)
dtree_max_depth_entry.insert(0, "10")  # Varsayılan değer
dtree_max_depth_entry.grid(row=9, column=1, padx=10)

tk.Label(model_frame, text="DTree Min Samples Split (Range: 2-20):").grid(row=9, column=2)
dtree_min_samples_split_entry = tk.Entry(model_frame, width=7)
dtree_min_samples_split_entry.insert(0, "2")  # Varsayılan değer
dtree_min_samples_split_entry.grid(row=9, column=3, padx=10)

step4_dtree_button = tk.Button(model_frame, text="DTree Modeli - Accuracy Göster", command=step4_dtree_optimized)
step4_dtree_button.grid(row=10, column=0, padx=5)

# Yeni eklenen buton
step4_dtree_optimize_button = tk.Button(model_frame, text="DTree Hiperparametre Optimize Et", command=optimize_dtree_hyperparameters)
step4_dtree_optimize_button.grid(row=10, column=1, padx=5)

# MRI görüntülerinin seçimi için ComboBox ve yansıtma butonunu ekleme
selection_frame = tk.Frame(root)
selection_frame.pack(pady=10)

# ComboBox ve Yansıt butonu global olarak tanımlanıyor
combobox = ttk.Combobox(selection_frame)
combobox.grid(row=0, column=0, padx=10)

reflect_button = tk.Button(selection_frame, text="Yansıt", command=display_selected_mri)
reflect_button.grid(row=0, column=1, padx=10)

show_plots_button = tk.Button(root, text="Grafikleri Göster", command=show_plots)
show_plots_button.pack(pady=10)

root.mainloop()
