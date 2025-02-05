import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans



# Load the dataset
file_path = '~/Documents/Data sets/Spotify-2000.csv'
df = pd.read_csv(file_path)

# Preprocess the data
# Handle missing values
df = df.dropna()


# Load the dataset
file_path = '~/Documents/Data sets/Spotify-2000.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Display a summary of the dataset
print(df.describe())

# Display the column names to identify the features
print(df.columns)

## 1. Chosing my features

# Feature selection
features = ['Energy','Danceability','Liveness','Valence','Acousticness','Speechiness' ] 
X = df[features]


## 2. Visualizing the data

# Plot the distribution of each variable
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualizing the normalized data (optional)
normalized_df = pd.DataFrame(X_scaled, columns=features)
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(normalized_df[feature], kde=True)
    plt.title(f'Normalized Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


## 3. Principal Component Analysis

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Visualize the coefficients of the first two principal components
plt.figure(figsize=(10, 6))
components = pd.DataFrame(pca.components_, columns=features, index=['Principal Component 1', 'Principal Component 2'])
sns.heatmap(components.T, annot=True, cmap='coolwarm')
plt.title('PCA Coefficients')
plt.xlabel('Principal Components')
plt.ylabel('Features')
plt.show()

# Visualize the first two principal components in a 2D scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', data=pca_df)
plt.title('2D Scatter Plot of the First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Identify the songs with the highest and lowest values for the 1st principal component
highest_pc1_index = pca_df['Principal Component 1'].idxmax()
lowest_pc1_index = pca_df['Principal Component 1'].idxmin()

highest_pc1_song = df.iloc[highest_pc1_index]
lowest_pc1_song = df.iloc[lowest_pc1_index]

print("Song with the highest value for the 1st principal component:")
print(highest_pc1_song)
print("\nSong with the lowest value for the 1st principal component:")
print(lowest_pc1_song)

# Identify the songs with the highest and lowest values for the 2nd principal component
highest_pc2_index = pca_df['Principal Component 2'].idxmax()
lowest_pc2_index = pca_df['Principal Component 2'].idxmin()

highest_pc2_song = df.iloc[highest_pc2_index]
lowest_pc2_song = df.iloc[lowest_pc2_index]

print("\nSong with the highest value for the 2nd principal component:")
print(highest_pc2_song)
print("\nSong with the lowest value for the 2nd principal component:")
print(lowest_pc2_song)

## 4. Recommendation System

# Function to recommend songs using Euclidean distance
def recommend_songs(index_value, num_recommendations=5):
    # Find the row index for the given 'Index' column value
    song_index = df[df['Index'] == index_value].index[0]
    
    # Calculate the Euclidean distances between the selected song and all other songs
    distances = cdist(X_scaled[song_index].reshape(1, -1), X_scaled, metric='euclidean').flatten()
    
    # Get the indices of the closest songs (excluding the selected song itself)
    closest_indices = distances.argsort()[1:num_recommendations+1]
    
    # Print the input song details
    input_song = df.iloc[song_index]
    print(f"Input Song: {input_song['Title']} by {input_song['Artist']}\n")
    
    # Print the recommended songs
    print("Recommended Songs:")
    for idx in closest_indices:
        recommended_song = df.iloc[idx]
        print(f"{recommended_song['Title']} by {recommended_song['Artist']}")
    
    # Return the recommended songs
    return df.iloc[closest_indices]

# Test the recommendation engine
sample_song_index_value = 10  # Replace with the 'Index' value of the song you want to use as input
recommendations = recommend_songs(sample_song_index_value)
print(recommendations)


## 5. Extra Work

## 5.A. Incorporating Popularity

# Function to recommend songs using Euclidean distance
def recommend_songs(index_value, num_recommendations=5):
    # Find the row index for the given 'Index' column value
    song_index = df[df['Index'] == index_value].index[0]
    
    # Calculate the Euclidean distances between the selected song and all other songs
    distances = cdist(X_scaled[song_index].reshape(1, -1), X_scaled, metric='euclidean').flatten()
    
    # Get the indices of the closest songs (excluding the selected song itself)
    closest_indices = distances.argsort()[1:num_recommendations+1]
    
    # Print the input song details
    input_song = df.iloc[song_index]
    print(f"Input Song: {input_song['Title']} by {input_song['Artist']}\n")
    
    # Print the recommended songs
    print("Recommended Songs:")
    recommended_songs = df.iloc[closest_indices]
    recommended_songs = recommended_songs.sort_values(by='Popularity', ascending=False)
    for idx, song in recommended_songs.iterrows():
        print(f"{song['Title']} by {song['Artist']} (Popularity: {song['Popularity']})")
    
    # Return the recommended songs
    return recommended_songs

# Test the recommendation engine
sample_song_index_value = 10  # Replace with the 'Index' value of the song you want to use as input
recommendations = recommend_songs(sample_song_index_value)
print(recommendations)

## 5.B. PCA from Scratch
import numpy as np

def pca_from_scratch(X, n_components=2):
    # Standardize the data
    X_meaned = X - np.mean(X, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(X_meaned, rowvar=False)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Select the top n_components eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    
    # Project the data onto principal components
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    
    return X_reduced

# Perform PCA from scratch
X_reduced_scratch = pca_from_scratch(X_scaled, n_components=2)
print(X_reduced_scratch)

# 5.C. Clustering
# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = kmeans.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_reduced_scratch[:, 0], y=X_reduced_scratch[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Determine the optimal number of clusters using the elbow method
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()