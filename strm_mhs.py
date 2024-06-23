# import Library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import altair as alt

# judul/Title web
st.title('Clustering Data Mahasiswa')
st.markdown("""
### ini merupakan clustering web berbasis Machine Learning untuk clustering data mahasiswa
### Dibuat oleh: kelompok 5
### Kecerdasan Bisnis B
Anggota: \n
         1. Daffa Ardiyansyah 21_134 \n
    2. Nurfaida Oktafinai 21_078  
         

________________________________________________________________________________________________
            
""")

# Baca Dataset
st.header("Membaca Dataset !!")
df = pd.read_excel('pre_mhs.xlsx')

# Menampilkan Data
st.subheader("Menampilkan Data Awal")
st.write(df)


# pre-procesing
st.header("Pre-procesing")
st.subheader('Seleksi Fitur')
# Selecting specific columns
dfOriginal = df[['gender', 'asal_sma', 'kotaasal', 'prodi']]
st.write(dfOriginal)

# Ubah data menjadi Numerik
st.subheader('Ubah Data Menjadi Numerik / LabelEncoder')
# Step 1: Encode categorical variables
le_gender = LabelEncoder()
le_asal_sma = LabelEncoder()
le_kotaasal = LabelEncoder()
le_prodi = LabelEncoder()

dfOriginal['gender_encoded'] = le_gender.fit_transform(dfOriginal['gender'])
dfOriginal['asal_sma_encoded'] = le_asal_sma.fit_transform(dfOriginal['asal_sma'])
dfOriginal['kotaasal_encoded'] = le_kotaasal.fit_transform(dfOriginal['kotaasal'])
dfOriginal['prodi_encoded'] = le_prodi.fit_transform(dfOriginal['prodi'])

# Gunakan data yang sudah di-encode untuk clustering
dfModel = dfOriginal[['gender_encoded', 'asal_sma_encoded', 'kotaasal_encoded', 'prodi_encoded']]

# data untuk Euclidien
EucModel = dfOriginal[['gender_encoded', 'asal_sma_encoded', 'kotaasal_encoded', 'prodi_encoded']]
AhcEucModel = dfOriginal[['gender_encoded', 'asal_sma_encoded', 'kotaasal_encoded', 'prodi_encoded']]
# data untuk Manhattan
ManModel = dfOriginal[['gender_encoded', 'asal_sma_encoded', 'kotaasal_encoded', 'prodi_encoded']]
# data untuk Minkowski
MinModel = dfOriginal[['gender_encoded', 'asal_sma_encoded', 'kotaasal_encoded', 'prodi_encoded']]

st.write(dfModel)



st.header('Melihat cluster terbaik dari data yang ada !')

# Menentukan Cluster terbaik dengan metode Elbow Curve
# Tentukan rentang jumlah cluster yang akan diuji
range_n_clusters = range(2, 10)

# List untuk menyimpan nilai inertia untuk setiap jumlah cluster
inertia = []

# Looping untuk setiap nilai n_clusters
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(dfModel)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertia, marker='o')
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range_n_clusters)
plt.grid(True)
# Tampilkan plot menggunakan st.pyplot()
st.pyplot()

# Menentukan Cluster terbaik dengan metode Silhouette Score
# List untuk menyimpan nilai silhouette score untuk setiap jumlah cluster
silhouette_avg = []

# Looping untuk setiap nilai n_clusters
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(dfModel)
    silhouette_avg.append(silhouette_score(dfModel, cluster_labels))

# Plot Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range_n_clusters)
plt.grid(True)
# Tampilkan plot menggunakan st.pyplot()
st.pyplot()


# Modeling - Menentukan cluster
st.sidebar.subheader("Tentukan Nilai K")
clust = st.sidebar.slider("Pilih Jumlah CLuster :", 2,10,3,1)

# Model
def model(n_clust):
    # Euclidien K-Means
    kmeans_euclidean = KMeans(n_clusters=n_clust, random_state=42)
    kmeans_euclidean.fit(EucModel)
    EucModel['Cluster'] = kmeans_euclidean.labels_

    # Inisialisasi AHC dengan Euclidien
    agglom = AgglomerativeClustering(n_clusters=n_clust, metric='euclidean', linkage='complete')
    # Fit the model
    agglom.fit(AhcEucModel)
    ahcLabels = agglom.labels_


    # Menampilkan Output
    st.header("Menampilkan Output")

    st.subheader("Menampilkan nilai sering muncul tiap cluster Kmeans")
    # Menampilkan MAX Kmeans
    # Mengembalikan hasil clustering ke teks asli
    dfkmeans= dfOriginal.copy()
    dfkmeans['Cluster'] = EucModel['Cluster']
    
    # Fungsi untuk mendekode nilai yang telah di-encode
    def decode_column(series, le):
        return le.inverse_transform(series)
    
    # Decode the encoded columns back to original text
    dfkmeans['gender'] = decode_column(dfkmeans['gender_encoded'], le_gender)
    dfkmeans['asal_sma'] = decode_column(dfkmeans['asal_sma_encoded'], le_asal_sma)
    dfkmeans['kotaasal'] = decode_column(dfkmeans['kotaasal_encoded'], le_kotaasal)
    dfkmeans['prodi'] = decode_column(dfkmeans['prodi_encoded'], le_prodi)
    
    # Fungsi untuk mendapatkan nilai paling banyak (mode) dalam setiap cluster
    def get_cluster_modes(df, cluster_col, target_col):
        return df.groupby(cluster_col)[target_col].agg(lambda x: x.mode()[0])
    
    # Mendapatkan nilai paling banyak (mode) untuk setiap atribut dalam setiap cluster
    mode_gender = get_cluster_modes(dfkmeans, 'Cluster', 'gender')
    mode_asal_sma = get_cluster_modes(dfkmeans, 'Cluster', 'asal_sma')
    mode_kotaasal = get_cluster_modes(dfkmeans, 'Cluster', 'kotaasal')
    mode_prodi = get_cluster_modes(dfkmeans, 'Cluster', 'prodi')
    
    # Menghitung ukuran setiap cluster
    cluster_sizes = dfkmeans['Cluster'].value_counts().sort_index()
    
    # Menampilkan hasil
    st.write("Mode values for each cluster: K-MEANS")
    
    for cluster in mode_gender.index:
        st.write('--------------------')
        st.write(f"\nCluster {cluster}:")
        st.write(f"  Banyak Data: {cluster_sizes.loc[cluster]}")
        st.write(f"  Gender: {mode_gender.loc[cluster]}")
        st.write(f"  Asal SMA: {mode_asal_sma.loc[cluster]}")
        st.write(f"  Kota Asal: {mode_kotaasal.loc[cluster]}")
        st.write(f"  Prodi: {mode_prodi.loc[cluster]}")


    st.subheader("Menampilkan nilai sering muncul tiap cluster AHC")
    # Add AHC cluster labels to the original DataFrame
    dfahc = dfOriginal.copy()
    dfahc['Cluster'] = ahcLabels

    # Function to decode encoded columns back to original text
    def decode_column(series, le):
        return le.inverse_transform(series)

    # Decode the encoded columns back to original text
    dfahc['gender'] = decode_column(dfahc['gender_encoded'], le_gender)
    dfahc['asal_sma'] = decode_column(dfahc['asal_sma_encoded'], le_asal_sma)
    dfahc['kotaasal'] = decode_column(dfahc['kotaasal_encoded'], le_kotaasal)
    dfahc['prodi'] = decode_column(dfahc['prodi_encoded'], le_prodi)

    # Function to get the mode value in each cluster
    def get_cluster_modes(df, cluster_col, target_col):
        return df.groupby(cluster_col)[target_col].agg(lambda x: x.mode()[0])

    # Get mode values for each attribute in each cluster
    mode_gender = get_cluster_modes(dfahc, 'Cluster', 'gender')
    mode_asal_sma = get_cluster_modes(dfahc, 'Cluster', 'asal_sma')
    mode_kotaasal = get_cluster_modes(dfahc, 'Cluster', 'kotaasal')
    mode_prodi = get_cluster_modes(dfahc, 'Cluster', 'prodi')

    # Calculate the size of each cluster
    cluster_sizes = dfahc['Cluster'].value_counts().sort_index()

    # Display the results
    st.write("Mode values for each cluster: AHC")

    for cluster in mode_gender.index:
        st.write('--------------------')
        st.write(f"\nCluster {cluster}:")
        st.write(f"  Banyak Data: {cluster_sizes.loc[cluster]}")
        st.write(f"  Gender: {mode_gender.loc[cluster]}")
        st.write(f"  Asal SMA: {mode_asal_sma.loc[cluster]}")
        st.write(f"  Kota Asal: {mode_kotaasal.loc[cluster]}")
        st.write(f"  Prodi: {mode_prodi.loc[cluster]}")


    # Scatter plot dengan hasil clustering
    # KMeans
    st.subheader('KMeans Clustering with Euclidean Distance')
    # Buat scatter plot interaktif dengan Altair
    scatter_plot = alt.Chart(dfkmeans).mark_circle(size=60).encode(
        x='asal_sma_encoded',
        y='prodi_encoded',
        color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10')),  # Menggunakan skema warna yang berbeda
        tooltip=['prodi','kotaasal','asal_sma'], # Tambahkan prodi sebagai tooltip
    ).properties(
        width=800,
        height=500,
        title='KMeans Clustering Scater Plot'
    )

    # Tampilkan plot menggunakan Streamlit
    st.altair_chart(scatter_plot)

    # KMeans
    st.subheader('AHC Clustering with Euclidean Distance')
    # Buat scatter plot interaktif dengan Altair
    scatter_plot = alt.Chart(dfahc).mark_circle(size=60).encode(
        x='asal_sma_encoded',
        y='prodi_encoded',
        color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10')),  # Menggunakan skema warna yang berbeda
        tooltip=['prodi','kotaasal','asal_sma'], # Tambahkan prodi sebagai tooltip
    ).properties(
        width=800,
        height=500,
        title='AHC Clustering Scater Plot'
    )

    # Tampilkan plot menggunakan Streamlit
    st.altair_chart(scatter_plot)

    st.header("Cek Akurasi")
    st.subheader("KMeans")
    st.write('Hitung silhouette score') 
    # Contoh penggunaan dengan membatasi ukuran sampel
    silhouette_avg = silhouette_score(EucModel, EucModel['Cluster'])
    st.write(f"Silhouette Score for K-Means clustering : {silhouette_avg}")

    st.write('Hitung Davies-Bouldin Index')
    dbi_score = davies_bouldin_score(EucModel, EucModel['Cluster'])
    st.write(f"Davies-Bouldin Index for K-Means clustering: {dbi_score}")

    st.subheader("AHC")
    st.write('Hitung silhouette score')
    # Contoh penggunaan dengan membatasi ukuran sampel
     # Misalnya, gunakan sampel sebesar 8000 data
    silhouette_avg_ahc = silhouette_score(AhcEucModel, ahcLabels)
    st.write(f"Silhouette Score for Agglomerative Clustering : {silhouette_avg_ahc}")

    st.write('Hitung Davies-Bouldin Index')
    dbi_score_ahc = davies_bouldin_score(AhcEucModel, ahcLabels)
    st.write(f"Davies-Bouldin Index for Agglomerative Clustering: {dbi_score_ahc}")

# Menjalankan Model
model(clust)