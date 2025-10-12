import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="💳",
    layout="wide"
)

API_URL = "https://api-scoring-credit-final.onrender.com"

try:
    response = requests.get(f"{API_URL}/status", timeout=2)
    api_status = response.json()
    api_ok = api_status['status'] == 'operational'
except:
    api_ok = False

st.title("💳 Dashboard Scoring Crédit - Prêt à Dépenser")

if not api_ok:
    st.error("⚠️ API non accessible. Lancez l'API avec: uvicorn main:app --reload")
    st.stop()

try:
    model_info = requests.get(f"{API_URL}/model/info").json()
    st.success(f"✓ Modèle: {model_info['model_type'].upper()} | AUC: {model_info['auc_score']:.4f} | Coût: {model_info['optimal_cost']:,}€ | Seuil: {model_info['optimal_threshold']:.1%}")
except:
    st.warning("Impossible de charger les infos du modèle")

@st.cache_data
def load_test_data():
    try:
        df = pd.read_csv("../all_clients_test.csv")
        return df
    except Exception as e:
        st.error(f"Erreur chargement données: {e}")
        return None

test_clients = load_test_data()

if test_clients is None:
    st.error("Impossible de charger all_clients_test.csv")
    st.stop()

st.sidebar.header("🔍 Recherche Client")

client_ids = sorted(test_clients['SK_ID_CURR'].unique())
selected_client_id = st.sidebar.selectbox(
    "Choisir un client (SK_ID_CURR)",
    options=client_ids,
    index=0
)

st.sidebar.markdown(f"**{len(client_ids)} clients disponibles dans le test set**")

if st.sidebar.button("📊 Analyser", type="primary", use_container_width=True):
    with st.spinner("Analyse en cours..."):
        try:
            client_row = test_clients[test_clients['SK_ID_CURR'] == selected_client_id].iloc[0]
            
            features_to_drop = ['SK_ID_CURR', 'RISK_SCORE', 'DECISION', 'REAL_TARGET']
            features = client_row.drop(features_to_drop).values.tolist()
            
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.header(f"Client SK_ID_CURR: {selected_client_id}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risk = result['risk_score']
                    st.metric("Risque de Défaut", f"{risk:.2%}")
                
                with col2:
                    decision = result['decision']
                    color = "🟢" if decision == "ACCORD" else "🔴"
                    st.metric("Décision", f"{color} {decision}")
                
                with col3:
                    st.metric("Seuil du Modèle", f"{result['threshold']:.1%}")
                
                with col4:
                    real_target = int(client_row['REAL_TARGET'])
                    real_label = "Défaut" if real_target == 1 else "Bon"
                    st.metric("Réalité", real_label)
                
                is_correct = (decision == "ACCORD" and real_target == 0) or (decision == "REFUS" and real_target == 1)
                
                if not is_correct:
                    st.warning("⚠️ Prédiction incorrecte")
                else:
                    st.success("✓ Prédiction correcte")
                
                st.subheader("📊 Niveau de Risque")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk * 100,
                    title={'text': "Risque de Défaut (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if risk >= result['threshold'] else "darkgreen"},
                        'steps': [
                            {'range': [0, result['threshold']*100], 'color': "lightgreen"},
                            {'range': [result['threshold']*100, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': result['threshold'] * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🔍 Facteurs de Décision (SHAP)")
                explain_resp = requests.post(f"{API_URL}/explain", json={"features": features}, timeout=10)
                
                if explain_resp.status_code == 200:
                    explanation = explain_resp.json()
                    shap_df = pd.DataFrame(explanation["top_features"])
                    
                    fig_shap = px.bar(
                        shap_df.sort_values("impact", key=abs),
                        x="impact",
                        y="feature",
                        orientation='h',
                        color="direction",
                        color_discrete_map={"AUGMENTE LE RISQUE": "red", "DIMINUE LE RISQUE": "green"},
                        labels={"impact": "Impact SHAP", "feature": "Variable"},
                        title="Top 10 Facteurs Influençant la Décision"
                    )
                    fig_shap.update_layout(height=500, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    st.info("💡 " + explanation["interpretation"])
                else:
                    st.warning("Explication SHAP non disponible")
                
                st.subheader("💡 Interprétation")
                if decision == "ACCORD":
                    st.success(f"""
                    **Crédit Accordé**
                    - Risque: {risk:.2%} < Seuil: {result['threshold']:.1%}
                    - Profil acceptable pour l'octroi du crédit
                    """)
                else:
                    st.error(f"""
                    **Crédit Refusé**
                    - Risque: {risk:.2%} ≥ Seuil: {result['threshold']:.1%}
                    - Profil à risque trop élevé
                    """)
                    
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

st.markdown("---")
st.markdown("**Projet 7 - OpenClassrooms** | Modèle LightGBM Champion | Test Set")