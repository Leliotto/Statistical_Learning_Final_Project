# %% [markdown]
# # Analisi Predittiva del Diabete - Statistical Learning in Epidemiologia
# 
# **Obiettivo**: Sviluppare un modello predittivo per il diabete mellito tipo 2 usando dati epidemiologici
# 
# **Dataset**: 403 soggetti afroamericani, Virginia - screening per diabete e fattori di rischio cardiovascolari
# 
# **Target**: Diagnosi diabete (glyhb > 7.0)

# %% [markdown]
# ## 1. SETUP & DATA LOADING

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, brier_score_loss, roc_curve,
                             precision_recall_curve,
                             classification_report, confusion_matrix)
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Statistiche
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configurazione plot
plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)

print("Librerie caricate con successo")

# %%
# Caricamento dataset
df = pd.read_csv("diabetes.csv")

print("=== INFORMAZIONI DATASET ===")
print(f"Shape: {df.shape}")
print(f"Variabili: {list(df.columns)}")
print(f"\nPrime 3 osservazioni:")
df.head(3)

# %% [markdown]
# ## 2. EXPLORATORY DATA ANALYSIS

# %%
# Analisi valori mancanti
print("=== ANALISI VALORI MANCANTI ===")
missing_info = df.isnull().sum()
missing_pct = (missing_info / len(df) * 100).round(1)
missing_df = pd.DataFrame({
    'Missing_Count': missing_info,
    'Missing_Percent': missing_pct
}).sort_values('Missing_Count', ascending=False)

print(missing_df[missing_df['Missing_Count'] > 0])

# Visualizzazione missing pattern
plt.figure(figsize=(10, 6))
plt.bar(range(len(missing_df[missing_df['Missing_Count'] > 0])), 
        missing_df[missing_df['Missing_Count'] > 0]['Missing_Percent'])
plt.xticks(range(len(missing_df[missing_df['Missing_Count'] > 0])), 
           missing_df[missing_df['Missing_Count'] > 0].index, rotation=45)
plt.ylabel('Percentuale Missing (%)')
plt.title('Pattern Valori Mancanti')
plt.tight_layout()
plt.show()

# %%
# Statistiche descrittive target variable
print("=== ANALISI VARIABILE TARGET (glyhb) ===")
glyhb_complete = df[~df['glyhb'].isnull()]['glyhb']

print(f"Osservazioni con glyhb valido: {len(glyhb_complete)}")
print(f"Range: {glyhb_complete.min():.2f} - {glyhb_complete.max():.2f}")
print(f"Media: {glyhb_complete.mean():.2f}")
print(f"Mediana: {glyhb_complete.median():.2f}")

# Prevalenza diabete con soglia 7.0
diabetes_cases = (glyhb_complete > 7.0).sum()
total_cases = len(glyhb_complete)
prevalence = diabetes_cases / total_cases

print(f"\nDiagnosi diabete (glyhb > 7.0):")
print(f"  Diabetici: {diabetes_cases}")
print(f"  Non diabetici: {total_cases - diabetes_cases}")
print(f"  Prevalenza: {prevalence:.3f} ({prevalence*100:.1f}%)")

# Distribuzione glyhb
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(glyhb_complete, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(7.0, color='red', linestyle='--', linewidth=2, label='Soglia diabete (7.0)')
plt.xlabel('Emoglobina Glicata (glyhb)')
plt.ylabel('Frequenza')
plt.title('Distribuzione Emoglobina Glicata')
plt.legend()

plt.subplot(1, 2, 2)
diabetes_labels = ['Non Diabetici', 'Diabetici']
diabetes_counts = [total_cases - diabetes_cases, diabetes_cases]
plt.pie(diabetes_counts, labels=diabetes_labels, autopct='%1.1f%%')
plt.title('Prevalenza Diabete')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. PREPROCESSING PIPELINE

# %%
print("=== PREPROCESSING PIPELINE ===")

# STEP 1: Rimozione osservazioni con glyhb mancante
df_clean = df[~df['glyhb'].isnull()].copy()
print(f"Dopo rimozione glyhb missing: {df_clean.shape[0]} osservazioni")

# STEP 2: Creazione variabile target
df_clean['diagnosis'] = (df_clean['glyhb'] > 7.0).astype(int)

# STEP 3: Rimozione definitiva bp.2s e bp.2d (65% missing)
df_clean = df_clean.drop(columns=['bp.2s', 'bp.2d'])
print("Rimosse bp.2s e bp.2d (65% missing)")

# STEP 4: Rinominare colonne problematiche
df_clean = df_clean.rename(columns={'stab.glu': 'stab_glu'})

# STEP 5: Feature engineering
# BMI
df_clean['bmi'] = (df_clean['weight'] * 0.453592) / ((df_clean['height'] * 0.0254) ** 2)

# Waist-Hip Ratio
df_clean['whr'] = df_clean['waist'] / df_clean['hip']

# Età categorica (fasce clinicamente rilevanti)
df_clean['age_group'] = pd.cut(df_clean['age'], 
                              bins=[0, 40, 50, 60, 100], 
                              labels=['<40', '40-50', '50-60', '60+'])

# BMI categorico (WHO classification)
df_clean['bmi_category'] = pd.cut(df_clean['bmi'], 
                                 bins=[0, 18.5, 25, 30, 100],
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

print("Feature engineering completato: BMI, WHR, age_group, bmi_category")

# %%
# STEP 6: Selezione variabili per modeling e rimozione outliers
modeling_vars = ['chol', 'age', 'stab_glu', 'hdl', 'ratio', 'bmi', 'whr', 'bp.1s', 'bp.1d']

# Complete case analysis
df_modeling = df_clean[modeling_vars + ['diagnosis']].dropna()
print(f"Dopo complete case analysis: {df_modeling.shape[0]} osservazioni")

# Rimozione outliers (Z-score > 3)
print("\n=== RIMOZIONE OUTLIERS ===")
outliers_removed = 0

for col in modeling_vars:
    z_scores = np.abs(stats.zscore(df_modeling[col]))
    outliers = (z_scores > 3).sum()
    if outliers > 0:
        print(f"{col}: {outliers} outliers rimossi")
        df_modeling = df_modeling[z_scores <= 3]
        outliers_removed += outliers

print(f"Totale outliers rimossi: {outliers_removed}")
print(f"Dataset finale: {df_modeling.shape[0]} osservazioni")

# Verifica prevalenza finale
final_prevalence = df_modeling['diagnosis'].mean()
print(f"Prevalenza finale: {final_prevalence:.3f}")

# %%
# Dataset finale per modeling
X_final = df_modeling[modeling_vars]
y_final = df_modeling['diagnosis']

# Train/Test split stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, 
    test_size=0.2, 
    stratify=y_final,
    random_state=42
)

print("=== SPLIT TRAIN/TEST ===")
print(f"Training set: {X_train.shape[0]} osservazioni")
print(f"Test set: {X_test.shape[0]} osservazioni")
print(f"Prevalenza train: {y_train.mean():.3f}")
print(f"Prevalenza test: {y_test.mean():.3f}")

# %% [markdown]
# ## 4. ANALISI DESCRITTIVA - ODDS RATIO

# %%
# Analisi stratificata per diagnosi
print("=== ANALISI DESCRITTIVA STRATIFICATA ===")

stratified_stats = X_train.groupby(y_train).agg(['mean', 'std', 'median']).round(2)

# Visualizzazione distribuzioni per variabili chiave
key_vars = ['age', 'stab_glu', 'bmi', 'whr', 'chol', 'hdl']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(key_vars):
    # Istogrammi separati per diagnosi
    non_diabetic = X_train[y_train == 0][var]
    diabetic = X_train[y_train == 1][var]
    
    axes[i].hist(non_diabetic, alpha=0.7, label='Non Diabetici', bins=20)
    axes[i].hist(diabetic, alpha=0.7, label='Diabetici', bins=20)
    
    # Mediane
    axes[i].axvline(non_diabetic.median(), color='blue', linestyle='--', alpha=0.8)
    axes[i].axvline(diabetic.median(), color='orange', linestyle='--', alpha=0.8)
    
    axes[i].set_title(f'{var}')
    axes[i].legend()

plt.suptitle('Distribuzioni per Stato Diabetico', fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Test statistici per differenze tra gruppi
print("=== TEST STATISTICI DIFFERENZE TRA GRUPPI ===")

test_results = []
for var in modeling_vars:
    group_0 = X_train[y_train == 0][var]
    group_1 = X_train[y_train == 1][var]
    
    # Mann-Whitney U test (non-parametrico)
    stat, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
    
    median_0 = group_0.median()
    median_1 = group_1.median()
    
    test_results.append({
        'Variable': var,
        'Median_NonDiabetic': median_0,
        'Median_Diabetic': median_1,
        'P_value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

test_df = pd.DataFrame(test_results)
print(test_df.round(4))

# %%
# Calcolo OR Crude e Adjusted con bootstrap
from sklearn.utils import resample

def compute_or_bootstrap(X, y, n_bootstraps=1000):
    """Calcola OR e CI con bootstrap per modello univariabile"""
    or_vals = []
    
    for _ in range(n_bootstraps):
        X_b, y_b = resample(X, y, random_state=None)
        
        if len(np.unique(y_b)) < 2:
            continue
            
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        
        try:
            model.fit(X_b, y_b)
            or_vals.append(np.exp(model.coef_[0][0]))
        except:
            continue
    
    if len(or_vals) < 100:
        return np.nan, np.nan, np.nan
    
    return np.mean(or_vals), np.percentile(or_vals, 2.5), np.percentile(or_vals, 97.5)

# OR Crude (univariabili)
print("=== CALCOLO OR CRUDE ===")
crude_results = []

for var in modeling_vars:
    X_single = X_train[[var]]
    or_val, ci_low, ci_high = compute_or_bootstrap(X_single, y_train)
    
    crude_results.append({
        'Variable': var,
        'OR_Crude': or_val,
        'CI_Lower': ci_low,
        'CI_Upper': ci_high,
        'Significant': 'Yes' if not (ci_low <= 1.0 <= ci_high) else 'No'
    })

crude_df = pd.DataFrame(crude_results)

# %%
# OR Adjusted (multivariabile)
print("=== CALCOLO OR ADJUSTED ===")

def compute_multivar_or_bootstrap(X, y, n_bootstraps=1000):
    """Calcola OR adjusted con bootstrap"""
    n_vars = X.shape[1]
    or_matrix = []
    
    for _ in range(n_bootstraps):
        X_b, y_b = resample(X, y, random_state=None)
        
        if len(np.unique(y_b)) < 2:
            continue
            
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        
        try:
            model.fit(X_b, y_b)
            or_matrix.append(np.exp(model.coef_[0]))
        except:
            continue
    
    if len(or_matrix) < 100:
        return [np.nan] * n_vars, [np.nan] * n_vars, [np.nan] * n_vars
    
    or_matrix = np.array(or_matrix)
    means = np.mean(or_matrix, axis=0)
    lower = np.percentile(or_matrix, 2.5, axis=0)
    upper = np.percentile(or_matrix, 97.5, axis=0)
    
    return means, lower, upper

# Calcolo OR adjusted
adj_means, adj_lower, adj_upper = compute_multivar_or_bootstrap(X_train, y_train)

adjusted_results = []
for i, var in enumerate(modeling_vars):
    adjusted_results.append({
        'Variable': var,
        'OR_Adjusted': adj_means[i],
        'CI_Lower': adj_lower[i],
        'CI_Upper': adj_upper[i],
        'Significant': 'Yes' if not (adj_lower[i] <= 1.0 <= adj_upper[i]) else 'No'
    })

adjusted_df = pd.DataFrame(adjusted_results)

# %%
# Tabella comparativa OR (Publication Ready)
combined_or = crude_df.merge(adjusted_df, on='Variable', suffixes=('_crude', '_adj'))

print("="*80)
print("                    ODDS RATIO RESULTS")
print("="*80)
print(f"{'Variable':<12} {'OR Crude (95% CI)':<20} {'OR Adjusted (95% CI)':<20} {'Confounding':<12}")
print("-"*80)

for _, row in combined_or.iterrows():
    var = row['Variable']
    
    # Crude
    crude_or = row['OR_Crude']
    crude_ci_low = row['CI_Lower_crude']
    crude_ci_up = row['CI_Upper_crude']
    crude_str = f"{crude_or:.2f} ({crude_ci_low:.2f}-{crude_ci_up:.2f})"
    
    # Adjusted
    adj_or = row['OR_Adjusted']
    adj_ci_low = row['CI_Lower_adj']
    adj_ci_up = row['CI_Upper_adj']
    adj_str = f"{adj_or:.2f} ({adj_ci_low:.2f}-{adj_ci_up:.2f})"
    
    # Confounding assessment
    if not np.isnan(crude_or) and not np.isnan(adj_or) and crude_or != 0:
        change = abs(crude_or - adj_or) / crude_or * 100
        confounding = "Yes" if change > 10 else "No"
    else:
        confounding = "N/A"
    
    print(f"{var:<12} {crude_str:<20} {adj_str:<20} {confounding:<12}")

print("="*80)

# %%
# Forest Plot per OR
def create_forest_plot(crude_df, adjusted_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    variables = crude_df['Variable'].tolist()
    y_positions = np.arange(len(variables))
    
    # Colori
    crude_color = '#2E86AB'
    adj_color = '#A23B72'
    
    for i, var in enumerate(variables):
        # Crude OR
        crude_or = crude_df.iloc[i]['OR_Crude']
        crude_ci_low = crude_df.iloc[i]['CI_Lower']
        crude_ci_high = crude_df.iloc[i]['CI_Upper']
        
        # Adjusted OR
        adj_or = adjusted_df.iloc[i]['OR_Adjusted']
        adj_ci_low = adjusted_df.iloc[i]['CI_Lower']
        adj_ci_high = adjusted_df.iloc[i]['CI_Upper']
        
        # Plot con offset verticale
        if not np.isnan(crude_or):
            ax.errorbar(crude_or, i + 0.1, 
                       xerr=[[crude_or - crude_ci_low], [crude_ci_high - crude_or]],
                       fmt='o', color=crude_color, capsize=5, capthick=2,
                       markersize=8, label='Crude OR' if i == 0 else "")
        
        if not np.isnan(adj_or):
            ax.errorbar(adj_or, i - 0.1,
                       xerr=[[adj_or - adj_ci_low], [adj_ci_high - adj_or]], 
                       fmt='s', color=adj_color, capsize=5, capthick=2,
                       markersize=8, label='Adjusted OR' if i == 0 else "")
    
    # Linea di riferimento OR = 1
    ax.axvline(1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Odds Ratio (log scale)', fontsize=12)
    ax.set_title('Forest Plot: Crude vs Adjusted Odds Ratios', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Annotazioni
    xlim = ax.get_xlim()
    ax.text(xlim[0]*1.1, len(variables), 'Protective', ha='left', fontsize=10, style='italic')
    ax.text(xlim[1]*0.9, len(variables), 'Risk Factor', ha='right', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()

create_forest_plot(crude_df, adjusted_df)

# %% [markdown]
# ## 5. PREDICTIVE MODELING

# %%
# Definizione modelli candidati (basati sui risultati OR)
models_config = {
    'Minimal': ['age', 'stab_glu'],
    'Metabolic': ['age', 'stab_glu', 'hdl', 'chol'], 
    'Anthropometric': ['age', 'bmi', 'whr'],
    'Complete': ['age', 'stab_glu', 'hdl', 'chol', 'bmi', 'whr']
}

# Aggiunta interazioni per modello completo
X_train_interactions = X_train.copy()
X_test_interactions = X_test.copy()

# Interazioni clinicamente rilevanti
X_train_interactions['age_bmi'] = X_train['age'] * X_train['bmi']
X_train_interactions['chol_hdl_ratio'] = X_train['chol'] / X_train['hdl']

X_test_interactions['age_bmi'] = X_test['age'] * X_test['bmi']
X_test_interactions['chol_hdl_ratio'] = X_test['chol'] / X_test['hdl']

models_config['Complete_Interactions'] = ['age', 'stab_glu', 'hdl', 'chol', 'bmi', 'whr', 
                                         'age_bmi', 'chol_hdl_ratio']

print("=== MODELLI CANDIDATI ===")
for name, vars in models_config.items():
    print(f"{name}: {len(vars)} variabili")

# %%
# Check multicollinearità
print("=== CONTROLLO MULTICOLLINEARITÀ ===")

def calculate_vif(X, variables):
    X_subset = X[variables]
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_subset.columns
    vif_data["VIF"] = [variance_inflation_factor(X_subset.values, i) 
                       for i in range(X_subset.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

for name, variables in models_config.items():
    if name == 'Complete_Interactions':
        X_check = X_train_interactions
    else:
        X_check = X_train
        
    # Verifica che tutte le variabili esistano
    available_vars = [v for v in variables if v in X_check.columns]
    if len(available_vars) != len(variables):
        print(f"⚠️ {name}: alcune variabili non disponibili")
        continue
        
    vif_results = calculate_vif(X_check, available_vars)
    high_vif = vif_results[vif_results['VIF'] > 5]
    
    if len(high_vif) > 0:
        print(f"\n{name} - Alta multicollinearità:")
        print(high_vif)

# %%
# Cross-validation con SMOTE
print("=== CROSS-VALIDATION CON SMOTE ===")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['roc_auc', 'precision', 'recall', 'f1']

cv_results = {}

for model_name, variables in models_config.items():
    print(f"\n🔍 {model_name} Model:")
    
    # Selezione dataset appropriato
    if model_name == 'Complete_Interactions':
        X_model_train = X_train_interactions[variables]
        X_model_test = X_test_interactions[variables]
    else:
        X_model_train = X_train[variables]
        X_model_test = X_test[variables]
    
    # Pipeline con SMOTE
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('logistic', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Cross-validation
    cv_scores = cross_validate(
        pipeline, X_model_train, y_train,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False
    )
    
    # Salva risultati
    cv_results[model_name] = {
        'auc': np.mean(cv_scores['test_roc_auc']),
        'auc_std': np.std(cv_scores['test_roc_auc']),
        'precision': np.mean(cv_scores['test_precision']),
        'recall': np.mean(cv_scores['test_recall']),
        'f1': np.mean(cv_scores['test_f1'])
    }
    
    print(f"  AUC: {cv_results[model_name]['auc']:.3f} ± {cv_results[model_name]['auc_std']:.3f}")
    print(f"  F1:  {cv_results[model_name]['f1']:.3f}")
    print(f"  Precision: {cv_results[model_name]['precision']:.3f}")
    print(f"  Recall: {cv_results[model_name]['recall']:.3f}")

# %%
# Selezione miglior modello e training finale
best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['auc'])
best_variables = models_config[best_model_name]

print(f"\n🏆 MIGLIOR MODELLO: {best_model_name}")
print(f"AUC: {cv_results[best_model_name]['auc']:.3f}")
print(f"Variabili: {best_variables}")

# Training modello finale
if best_model_name == 'Complete_Interactions':
    X_best_train = X_train_interactions[best_variables]
    X_best_test = X_test_interactions[best_variables]
else:
    X_best_train = X_train[best_variables]
    X_best_test = X_test[best_variables]

final_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('logistic', LogisticRegression(random_state=42, max_iter=1000))
])

final_pipeline.fit(X_best_train, y_train)

# %% [markdown]
# ## 6. MODEL VALIDATION & PERFORMANCE

# %%
# Predizioni su test set (PRIMA VOLTA che tocchiamo il test set!)
print("=== VALIDAZIONE SU TEST SET ===")

y_test_pred = final_pipeline.predict(X_best_test)
y_test_proba = final_pipeline.predict_proba(X_best_test)[:, 1]

# Metriche di performance
test_auc = roc_auc_score(y_test, y_test_proba)
test_brier = brier_score_loss(y_test, y_test_proba)

print(f"Test AUC: {test_auc:.3f}")
print(f"Test Brier Score: {test_brier:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Non-Diabetic', 'Diabetic']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# %%
# Visualizzazioni performance
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.3f})', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - Test Set')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
axes[1].plot(recall, precision, linewidth=2)
axes[1].axhline(y=y_test.mean(), color='red', linestyle='--', 
                label=f'Baseline (prevalence: {y_test.mean():.3f})')
axes[1].set_xlabel('Recall (Sensitivity)')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Calibration Plot
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_test_proba, n_bins=10)
axes[2].plot(mean_predicted_value, fraction_of_positives, "s-", 
             label=f'Model (Brier: {test_brier:.3f})')
axes[2].plot([0, 1], [0, 1], "k:", label="Perfect calibration")
axes[2].set_xlabel('Mean Predicted Probability')
axes[2].set_ylabel('Fraction of Positives')
axes[2].set_title('Calibration Plot')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Confronto modelli (Cross-validation results)
comparison_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'AUC': [cv_results[m]['auc'] for m in cv_results.keys()],
    'AUC_std': [cv_results[m]['auc_std'] for m in cv_results.keys()],
    'F1': [cv_results[m]['f1'] for m in cv_results.keys()],
    'Precision': [cv_results[m]['precision'] for m in cv_results.keys()],
    'Recall': [cv_results[m]['recall'] for m in cv_results.keys()]
}).sort_values('AUC', ascending=False)

print("=== CONFRONTO MODELLI (Cross-Validation) ===")
print(comparison_df.round(3))

# Visualizzazione confronto
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(comparison_df['Model'], comparison_df['AUC'], 
         xerr=comparison_df['AUC_std'], capsize=5)
plt.xlabel('AUC')
plt.title('Cross-Validation AUC Comparison')
plt.axvline(0.7, color='red', linestyle='--', alpha=0.7, label='Threshold')

plt.subplot(1, 2, 2)
plt.scatter(comparison_df['Precision'], comparison_df['Recall'], 
           s=comparison_df['AUC']*500, alpha=0.7)
for i, model in enumerate(comparison_df['Model']):
    plt.annotate(model, (comparison_df.iloc[i]['Precision'], comparison_df.iloc[i]['Recall']),
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall (size = AUC)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. INTERPRETAZIONE CLINICA

# %%
# Coefficienti del modello finale
print("=== INTERPRETAZIONE COEFFICIENTI MODELLO FINALE ===")

# Estrai il modello logistico dal pipeline
scaler = final_pipeline.named_steps['scaler']
logistic_model = final_pipeline.named_steps['logistic']

# Coefficienti e OR
coefficients = logistic_model.coef_[0]
odds_ratios = np.exp(coefficients)

# Crea tabella interpretativa
interpretation_df = pd.DataFrame({
    'Variable': best_variables,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios,
    'Interpretation': [''] * len(best_variables)
})

# Aggiungi interpretazioni
for i, row in interpretation_df.iterrows():
    or_val = row['Odds_Ratio']
    var_name = row['Variable']
    
    if or_val > 1.2:
        interp = f"Risk factor: +{(or_val-1)*100:.0f}% odds increase"
    elif or_val < 0.8:
        interp = f"Protective: -{(1-or_val)*100:.0f}% odds reduction"
    else:
        interp = "Neutral effect"
    
    interpretation_df.loc[i, 'Interpretation'] = interp

interpretation_df = interpretation_df.sort_values('Odds_Ratio', ascending=False)
print(interpretation_df.round(3))

# %%
# Feature importance plot
plt.figure(figsize=(10, 6))
abs_coefs = np.abs(interpretation_df['Coefficient'])
colors = ['red' if or_val > 1 else 'blue' for or_val in interpretation_df['Odds_Ratio']]

plt.barh(interpretation_df['Variable'], abs_coefs, color=colors, alpha=0.7)
plt.xlabel('|Coefficient| (Log Odds)')
plt.title('Feature Importance in Final Model')
plt.gca().invert_yaxis()

# Legenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='Risk Factor (OR > 1)'),
                  Patch(facecolor='blue', alpha=0.7, label='Protective Factor (OR < 1)')]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.show()

# %%
# Analisi predizioni per sottogruppi clinici
print("=== ANALISI PER SOTTOGRUPPI CLINICI ===")

# Aggiungi predizioni al test set
test_results = X_best_test.copy()
test_results['y_true'] = y_test
test_results['y_pred_proba'] = y_test_proba
test_results['y_pred'] = y_test_pred

# Categorizza per età e BMI
test_results['age_group'] = pd.cut(test_results['age'], 
                                  bins=[0, 50, 60, 100], 
                                  labels=['<50', '50-60', '60+'])

if 'bmi' in test_results.columns:
    test_results['bmi_group'] = pd.cut(test_results['bmi'], 
                                      bins=[0, 25, 30, 100],
                                      labels=['Normal', 'Overweight', 'Obese'])

# Performance per sottogruppi
if 'age_group' in test_results.columns:
    print("\nPerformance per gruppo di età:")
    for group in test_results['age_group'].unique():
        if pd.isna(group):
            continue
        mask = test_results['age_group'] == group
        group_auc = roc_auc_score(test_results[mask]['y_true'], 
                                 test_results[mask]['y_pred_proba'])
        n_patients = mask.sum()
        prevalence = test_results[mask]['y_true'].mean()
        print(f"  {group}: AUC={group_auc:.3f}, n={n_patients}, prevalence={prevalence:.3f}")

if 'bmi_group' in test_results.columns:
    print("\nPerformance per gruppo BMI:")
    for group in test_results['bmi_group'].unique():
        if pd.isna(group):
            continue
        mask = test_results['bmi_group'] == group
        if mask.sum() > 5:  # Solo se abbiamo abbastanza osservazioni
            group_auc = roc_auc_score(test_results[mask]['y_true'], 
                                     test_results[mask]['y_pred_proba'])
            n_patients = mask.sum()
            prevalence = test_results[mask]['y_true'].mean()
            print(f"  {group}: AUC={group_auc:.3f}, n={n_patients}, prevalence={prevalence:.3f}")

# %% [markdown]
# ## 8. SUMMARY & CONCLUSIONS

# %%
print("="*80)
print("                    DIABETES PREDICTION MODEL - SUMMARY")
print("="*80)

print(f"\n📊 DATASET:")
print(f"  • Total observations: {df.shape[0]}")
print(f"  • Final modeling dataset: {df_modeling.shape[0]} ({df_modeling.shape[0]/df.shape[0]*100:.1f}%)")
print(f"  • Training set: {len(y_train)} observations")
print(f"  • Test set: {len(y_test)} observations")
print(f"  • Diabetes prevalence: {y_final.mean():.1%}")

print(f"\n🔬 PREPROCESSING:")
print(f"  • Missing data: Complete case analysis")
print(f"  • Outliers: Removed (Z-score > 3)")
print(f"  • Class imbalance: Handled with SMOTE")
print(f"  • Feature engineering: BMI, WHR, interactions")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"  • Variables: {len(best_variables)}")
print(f"  • Cross-validation AUC: {cv_results[best_model_name]['auc']:.3f} ± {cv_results[best_model_name]['auc_std']:.3f}")
print(f"  • Test set AUC: {test_auc:.3f}")
print(f"  • Test set Brier Score: {test_brier:.3f}")

print(f"\n🎯 KEY FINDINGS:")
# Identifica i fattori di rischio più forti dal modello finale
strongest_risk_factors = interpretation_df[interpretation_df['Odds_Ratio'] > 1.2].sort_values('Odds_Ratio', ascending=False)
if len(strongest_risk_factors) > 0:
    print(f"  • Main risk factors:")
    for _, factor in strongest_risk_factors.head(3).iterrows():
        print(f"    - {factor['Variable']}: OR = {factor['Odds_Ratio']:.2f}")

protective_factors = interpretation_df[interpretation_df['Odds_Ratio'] < 0.8].sort_values('Odds_Ratio')
if len(protective_factors) > 0:
    print(f"  • Protective factors:")
    for _, factor in protective_factors.head(2).iterrows():
        print(f"    - {factor['Variable']}: OR = {factor['Odds_Ratio']:.2f}")

print(f"\n📈 MODEL PERFORMANCE:")
print(f"  • Discrimination: {'Good' if test_auc > 0.8 else 'Acceptable' if test_auc > 0.7 else 'Fair'} (AUC = {test_auc:.3f})")
print(f"  • Calibration: {'Good' if test_brier < 0.15 else 'Moderate' if test_brier < 0.25 else 'Poor'} (Brier = {test_brier:.3f})")

# Test performance summary
tn, fp, fn, tp = cm.flatten()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"  • Sensitivity (recall): {sensitivity:.3f}")
print(f"  • Specificity: {specificity:.3f}")
print(f"  • Positive Predictive Value: {ppv:.3f}")
print(f"  • Negative Predictive Value: {npv:.3f}")

print(f"\n💡 CLINICAL RELEVANCE:")
print(f"  • Model identifies {sensitivity*100:.0f}% of diabetic patients")
print(f"  • {(1-specificity)*100:.0f}% false positive rate")
print(f"  • Suitable for screening in similar populations")

print(f"\n⚠️  LIMITATIONS:")
print(f"  • Single-center study (African American population)")
print(f"  • Cross-sectional design")
print(f"  • Limited sample size for rare outcomes")
print(f"  • External validation needed")

print("="*80)

# %%
# Salvataggio risultati chiave
final_results = {
    'model_name': best_model_name,
    'variables': best_variables,
    'cv_auc': cv_results[best_model_name]['auc'],
    'cv_auc_std': cv_results[best_model_name]['auc_std'],
    'test_auc': test_auc,
    'test_brier': test_brier,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'ppv': ppv,
    'npv': npv,
    'coefficients': dict(zip(best_variables, coefficients)),
    'odds_ratios': dict(zip(best_variables, odds_ratios))
}

print("✅ Analisi completata con successo!")
print(f"Final model ready for deployment: {best_model_name}")
print(f"Expected AUC on new data: {test_auc:.3f}")

# %% [markdown]
# ---
# 
# ## Conclusioni
# 
# L'analisi ha sviluppato un modello predittivo per il diabete mellito tipo 2 con performance accettabili. Il modello finale ha identificato i principali fattori di rischio epidemiologici e può essere utilizzato per screening in popolazioni simili.
# 
# **Prossimi step suggeriti:**
# 1. Validazione esterna su dataset indipendente
# 2. Implementazione in ambiente clinico pilota
# 3. Studio prospettico per validazione longitudinale
# 4. Estensione ad altre popolazioni etniche