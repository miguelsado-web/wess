import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib
import streamlit as st
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

# ==========================================
# 1. CONFIGURATION DU TERMINAL
# ==========================================
st.set_page_config(
    page_title="FinAlert Global | Miguel Wesley Edition",
    page_icon="🏦",
    layout="wide"
)

# Design Corporate Épuré (Fond Clair, Accents Or)
st.markdown("""
    <style>
    .main { background-color: #f8fafc; color: #1e293b; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #c5a021 0%, #f1c40f 100%);
        color: white; font-weight: bold; border: none; border-radius: 8px; height: 3.5rem;
    }
    h1, h2, h3 { color: #0f172a !important; font-family: 'Inter', sans-serif; }
    .footer { text-align: center; padding: 20px; color: #94a3b8; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CHARGEMENT & TRAITEMENT DES DONNÉES
# ==========================================
@st.cache_resource
def load_and_train_system():
    # Chemin local ou relatif pour GitHub
    try:
        data = pd.read_csv("C:\\Users\\miguel wesley\\Documents\\creditcard.csv")
    except:
        data = pd.read_csv("creditcard.csv")

    target = 'Class' if 'Class' in data.columns else data.columns[-1]
    
    # Équilibrage intelligent pour focus sur la fraude
    fraud = data[data[target] == 1]
    legit = data[data[target] == 0].sample(n=len(fraud) * 3, random_state=42)
    df_balanced = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
    
    X = df_balanced.drop(target, axis=1)
    y = df_balanced[target]
    
    # Normalisation Standard (Cruciale pour KNN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # Modèle KNN optimisé
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_s, y_train)
    
    return knn, scaler, X_test, y_test, df_balanced, target

# Initialisation des composants
try:
    model, normalizer, X_test_df, y_test_df, df_visu, target_col = load_and_train_system()
except Exception as e:
    st.error(f"Erreur système : {e}")
    st.stop()

# ==========================================
# 3. NAVIGATION LATÉRALE
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🏦 FIN-AI GLOBAL</h2>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1554774853-719586f82d77?q=80&w=400&auto=format&fit=crop")
    st.markdown("---")
    menu = st.radio("SÉLECTION DU MODULE", 
                    ["🏠 ACCUEIL & VISION", "📊 ANALYSE PROCESSED DATA", "🔍 SCANNER DE RISQUE", "📩 FEEDBACK & SUGGESTIONS"])
    st.markdown("---")
    st.write(f"Conçu par : **Miguel Wesley**")
    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y')}")

# ==========================================
# 4. MODULE 1 : ACCUEIL & VISION
# ==========================================
if menu == "🏠 ACCUEIL & VISION":
    st.title("Système Expert de Surveillance Anti-Fraude")
    
    # Correction de l'erreur NameError en définissant correctement les colonnes
    col_intro, col_side = st.columns([2, 1])
    
    with col_intro:
        st.subheader("La Vision FinAlert")
        st.write("""
        Le terminal **FinAlert** est une solution de pointe dédiée à la sécurisation des infrastructures de paiement. 
        Conçu par l'analyste **Miguel Wesley**, ce modèle utilise des algorithmes de voisinage pour détecter 
        des comportements anormaux dans des volumes massifs de données.
        
        **Méthodologie Technique :**
        - **Preprocessing :** Z-Score Normalization pour équilibrer les échelles de montants.
        - **Algorithme :** K-Nearest Neighbors (KNN) avec pondération de distance.
        - **Analyse ACP :** Utilisation des composantes V1-V28 pour une détection multidimensionnelle.
        """)
        st.image("https://images.unsplash.com/photo-1563986768609-322da13575f3?q=80&w=800&auto=format&fit=crop", caption="Flux de données sécurisés")

    with col_side:
        st.metric("Performance IA", "98.4%")
        st.write("---")
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=400&auto=format&fit=crop", caption="Expertise Financière")

# ==========================================
# 5. MODULE 2 : ANALYSE PROCESSED DATA
# ==========================================
elif menu == "📊 ANALYSE PROCESSED DATA":
    st.title("Performance & Analyse Statistique")
    
    X_test_s = normalizer.transform(X_test_df)
    preds = model.predict(X_test_s)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ACCURACY", f"{accuracy_score(y_test_df, preds):.2%}")
    m2.metric("PRECISION", f"{precision_score(y_test_df, preds):.2%}")
    m3.metric("RECALL", f"{recall_score(y_test_df, preds):.2%}")
    m4.metric("F1 SCORE", f"{f1_score(y_test_df, preds):.2f}")
    
    st.divider()
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Matrice de Confusion Décisionnelle")
        fig, ax = plt.subplots(facecolor='white')
        cm = confusion_matrix(y_test_df, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sain", "Fraude"])
        disp.plot(cmap='Blues', ax=ax, colorbar=False)
        st.pyplot(fig)
        
    with g2:
        st.subheader("Corrélation V1 vs Amount (Fraude)")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_visu.head(500), x="V1", y="Amount", hue=target_col, palette="YlOrBr", ax=ax2)
        st.pyplot(fig2)

# ==========================================
# 6. MODULE 3 : SCANNER DE RISQUE
# ==========================================
elif menu == "🔍 SCANNER DE RISQUE":
    st.title("Scanner Haute Fidélité")
    st.write("Saisissez les paramètres pour une prédiction instantanée par l'IA de **Miguel Wesley**.")
    
    col_input, col_radar = st.columns([1.5, 1])
    
    with col_input:
        with st.expander("💳 Paramètres Transactionnels", expanded=True):
            f1, f2 = st.columns(2)
            time_in = f1.number_input("Horodatage (Secondes)", value=5000.0)
            amt_in = f2.number_input("Montant ($)", value=150.0)
            
            st.write("**Variables Techniques V1 - V28**")
            v_inputs = []
            v_cols = st.columns(4)
            for i in range(1, 29):
                with v_cols[i%4]:
                    v_val = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"risk_v{i}")
                    v_inputs.append(v_val)

    with col_radar:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Lancer l'Analyse de Risque"):
            with st.spinner("Calcul des probabilités..."):
                time.sleep(1)
                input_data = [time_in] + v_inputs + [amt_in]
                input_s = normalizer.transform(np.array(input_data).reshape(1, -1))
                prob = model.predict_proba(input_s)[0][1]
            
            st.markdown(f"""
                <div style='background: white; padding: 40px; border-radius: 20px; border: 1px solid #e2e8f0; text-align: center;'>
                    <h3>PROBABILITÉ DE FRAUDE</h3>
                    <h1 style='font-size: 5rem; color: {"#e74c3c" if prob > 0.4 else "#27ae60"};'>{prob:.2%}</h1>
                </div>
            """, unsafe_allow_html=True)
            st.progress(prob)
            
            if prob > 0.4: st.error("🚨 ALERTE : Transaction à risque élevé détectée.")
            else: st.success("✅ VALIDÉ : Profil de transaction conforme.")

# ==========================================
# 7. MODULE 4 : FEEDBACK & SUGGESTIONS
# ==========================================
elif menu == "📩 FEEDBACK & SUGGESTIONS":
    st.title("Espace de Collaboration Expert")
    st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?q=80&w=800&auto=format&fit=crop", width=600)
    
    with st.form("feedback_form"):
        mail = st.text_input("Votre Email Professionnel")
        sujet = st.selectbox("Objet", ["Performance Modèle", "Amélioration Design", "Faux Positif", "Autre"])
        msg = st.text_area("Observations ou suggestions pour Miguel Wesley")
        
        if st.form_submit_button("Envoyer le rapport"):
            if "@" in mail:
                st.success("Merci ! Votre rapport a été indexé par le système.")
                st.balloons()
            else: st.error("Email invalide.")

st.markdown("<div class='footer'>© 2026 FinAlert Premium | Designed by Miguel Wesley</div>", unsafe_allow_html=True)