import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

st.set_page_config(page_title="Titanic Full Dashboard", page_icon="🚢", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

# --- SIDEBAR ---
st.sidebar.image("https://i.imgur.com/yp4pvDn.png", width=160)
st.sidebar.title("🚢 Titanic Dashboard")
page = st.sidebar.radio("Sayfa Seç", ["Ana Sayfa", "Veri Keşfi", "EDA", "Scatter Plot", "Derived Score", "Correlation", "PCA & KMeans", "Random Forest", "Hakkında"])

# --- ANA SAYFA ---
if page == "Ana Sayfa":
    st.title("🚢 Titanic Yolcu Analizi")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Yolcu", len(df))
    col2.metric("Hayatta Kalan", df["Survived"].sum())
    col3.metric("Hayatta Kalma Oranı", f"%{df['Survived'].mean()*100:.1f}")
    col4.metric("Kadın Yolcu Oranı", f"%{(df['Sex']=='female').mean()*100:.1f}")
    st.markdown("---")
    colA, colB = st.columns(2)
    fig1 = px.histogram(df, x="Age", nbins=30, title="Yaş Dağılımı")
    fig2 = px.pie(df, names="Sex", title="Cinsiyet Dağılımı")
    colA.plotly_chart(fig1, width="stretch")
    colB.plotly_chart(fig2, width="stretch")

# --- VERİ KEŞFİ ---
elif page == "Veri Keşfi":
    st.title("📄 Veri Seti Keşfi")
    st.subheader("İlk 10 Satır")
    st.dataframe(df.head(10))
    st.subheader("Dataset Özeti")
    st.write(f"Satır: {df.shape[0]}, Sütun: {df.shape[1]}")
    st.write(f"Eksik değer: {df.isna().sum().sum()}")
    st.write(f"Kategorik sütunlar: {len(df.select_dtypes('object').columns)}")
    st.write("Sütun tipleri:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

# --- EDA (KATEGORİK ANALİZ) ---
elif page == "EDA":
    st.title("📊 Kategorik Değişken Analizi")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_col = st.selectbox("Kategorik sütun seçin", cat_cols)
    max_cat = st.slider("En fazla kaç kategori gösterilsin?", 2, 30, 5)
    plot_type = st.radio("Grafik tipi", ["Pie", "Bar"])

    counts = df[cat_col].value_counts().nlargest(max_cat)
    if plot_type == "Pie":
        fig = px.pie(values=counts.values, names=counts.index, title=f"{cat_col} dağılımı")
    else:
        fig = px.bar(x=counts.index, y=counts.values, title=f"{cat_col} dağılımı")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 10 kategori")
    top10_col = st.selectbox("Gruplama sütunu", cat_cols, key="top10")
    top_counts = df[top10_col].value_counts().nlargest(10)
    fig2 = px.bar(x=top_counts.index, y=top_counts.values, title=f"Top 10 {top10_col}")
    st.plotly_chart(fig2, width="stretch")

# --- SCATTER PLOT ---
elif page == "Scatter Plot":
    st.title("📈 Scatter Plot")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    x_col = st.selectbox("X ekseni", numeric_cols)
    y_col = st.selectbox("Y ekseni", numeric_cols, index=1)
    fig = px.scatter(df, x=x_col, y=y_col, color="Survived", title=f"{x_col} vs {y_col}")
    st.plotly_chart(fig, width="stretch")

# --- DERIVED SCORE ---
elif page == "Derived Score":
    st.title("💡 Derived Score (TotalScore)")
    score_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Seçilen sütunlar toplamı TotalScore olarak kullanılacak")
    selected_cols = st.multiselect("Seçin", score_cols, default=score_cols[:5])
    df["TotalScore"] = df[selected_cols].sum(axis=1)
    fig = px.box(df, x="Survived", y="TotalScore", title="TotalScore vs Survived")
    st.plotly_chart(fig, width="stretch")

# --- CORRELATION HEATMAP ---
elif page == "Correlation":
    st.title("🗺️ Korelasyon Heatmap")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- PCA & KMEANS ---
elif page == "PCA & KMeans":
    st.title("📉 PCA & KMeans Segmentasyonu")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_cols = st.multiselect("PCA için kullanılacak sütunlar", numeric_cols, default=numeric_cols[:5])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[selected_cols])
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=["PC1","PC2"])

    k = st.slider("Cluster sayısı (k)", 2, 6, 2)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(components)
    pca_df["Cluster"] = clusters.astype(str)

    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="KMeans Clusters")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Cluster Özetleri (feature ortalamaları)")
    cluster_summary = pd.DataFrame(df[selected_cols].groupby(clusters).mean())
    st.dataframe(cluster_summary)

# --- RANDOM FOREST ---
elif page == "Random Forest":
    st.title("🤖 Random Forest ile Hayatta Kalma Tahmini")
    df_model = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    numeric_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    all_features = numeric_cols + [col for col in df_model.columns if col.startswith("Sex_") or col.startswith("Embarked_")]
    test_size = st.slider("Test oranı", 0.1, 0.4, 0.2)
    n_estimators = st.slider("Ağaç sayısı (n_estimators)", 100, 600, 100)
    X = df_model[all_features]
    y = df_model["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.subheader("Sonuçlar")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Feature Importance")
    feat_imp = pd.DataFrame({"feature": all_features, "importance": rf.feature_importances_})
    feat_imp = feat_imp.sort_values(by="importance", ascending=False)
    fig = px.bar(feat_imp, x="feature", y="importance", title="Feature Importance")
    st.plotly_chart(fig, width="stretch")

# --- HAKKINDA ---
elif page == "Hakkında":
    st.title("ℹ️ Hakkında")
    st.markdown("""
    Bu proje Streamlit ile yapılmıştır.  
    Geliştirici: **Zehra Aras**  
    Tüm EDA, PCA, KMeans ve Random Forest özelliklerini içerir.
    """)
