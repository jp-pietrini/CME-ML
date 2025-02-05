<div style="text-align: center;">

# **CME250 HW1**
## Building a Spotify song through Content Based Filtering
### Juan Pablo Pietrini Sánchez

</div>

<p></p>
<p></p>

In this project we will build a song recommendation engine. The user will input one song and will receive five recommendations for songs to play.

# 1. Chosing the features

First we start by loading all the relevant libraries.


```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
```

Now we load the dataset we will use.


```python
# Load the dataset
file_path = '~/Documents/Data sets/Spotify-2000.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Display a summary of the dataset
print(df.describe())

# Display the column names to identify the features
print(df.columns)
```

       Index                   Title             Artist            Top Genre  \
    0      1                 Sunrise        Norah Jones      adult standards   
    1      2             Black Night        Deep Purple           album rock   
    2      3          Clint Eastwood           Gorillaz  alternative hip hop   
    3      4           The Pretender       Foo Fighters    alternative metal   
    4      5  Waitin' On A Sunny Day  Bruce Springsteen         classic rock   
    
       Year  Beats Per Minute (BPM)  Energy  Danceability  Loudness (dB)  \
    0  2004                     157      30            53            -14   
    1  2000                     135      79            50            -11   
    2  2001                     168      69            66             -9   
    3  2007                     173      96            43             -4   
    4  2002                     106      82            58             -5   
    
       Liveness  Valence Length (Duration)  Acousticness  Speechiness  Popularity  
    0        11       68               201            94            3          71  
    1        17       81               207            17            7          39  
    2         7       52               341             2           17          69  
    3         3       37               269             0            4          76  
    4        10       87               256             1            3          59  
                 Index         Year  Beats Per Minute (BPM)       Energy  \
    count  1994.000000  1994.000000             1994.000000  1994.000000   
    mean    997.500000  1992.992979              120.215647    59.679539   
    std     575.762538    16.116048               28.028096    22.154322   
    min       1.000000  1956.000000               37.000000     3.000000   
    25%     499.250000  1979.000000               99.000000    42.000000   
    50%     997.500000  1993.000000              119.000000    61.000000   
    75%    1495.750000  2007.000000              136.000000    78.000000   
    max    1994.000000  2019.000000              206.000000   100.000000   
    
           Danceability  Loudness (dB)     Liveness      Valence  Acousticness  \
    count   1994.000000    1994.000000  1994.000000  1994.000000   1994.000000   
    mean      53.238215      -9.008526    19.012036    49.408726     28.858074   
    std       15.351507       3.647876    16.727378    24.858212     29.011986   
    min       10.000000     -27.000000     2.000000     3.000000      0.000000   
    25%       43.000000     -11.000000     9.000000    29.000000      3.000000   
    50%       53.000000      -8.000000    12.000000    47.000000     18.000000   
    75%       64.000000      -6.000000    23.000000    69.750000     50.000000   
    max       96.000000      -2.000000    99.000000    99.000000     99.000000   
    
           Speechiness  Popularity  
    count  1994.000000  1994.00000  
    mean      4.994985    59.52658  
    std       4.401566    14.35160  
    min       2.000000    11.00000  
    25%       3.000000    49.25000  
    50%       4.000000    62.00000  
    75%       5.000000    71.00000  
    max      55.000000   100.00000  
    Index(['Index', 'Title', 'Artist', 'Top Genre', 'Year',
           'Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)',
           'Liveness', 'Valence', 'Length (Duration)', 'Acousticness',
           'Speechiness', 'Popularity'],
          dtype='object')


Finally we select the features we want to consider for the recommendation system, drop the missing values of the dataset, and normalize the values of the features so that they are presented in equivalent scales.


```python
# Handle missing values
df = df.dropna()

# Feature selection
features = ['Energy', 'Danceability', 'Liveness', 'Valence', 'Acousticness', 'Speechiness']
X = df[features]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

# 2. Visualizing the data


```python
# Plot the distribution of each variable
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Visualizing the normalized data (optional)
normalized_df = pd.DataFrame(X_scaled, columns=features)
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(normalized_df[feature], kde=True)
    plt.title(f'Normalized Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
```


    
![png](output_9_0.png)
    



    
![png](output_9_1.png)
    



    
![png](output_9_2.png)
    



    
![png](output_9_3.png)
    



    
![png](output_9_4.png)
    



    
![png](output_9_5.png)
    



    
![png](output_9_6.png)
    



    
![png](output_9_7.png)
    



    
![png](output_9_8.png)
    



    
![png](output_9_9.png)
    



    
![png](output_9_10.png)
    



    
![png](output_9_11.png)
    


As we can see the distribution of energy is slightly left-skewed, which show that most songs have between 60 and 80 of energy. Danceability follows a more normal distribution with a mean around 50. Liveness is strongly right-skewed, which could be interpreted as that most songs in the database don't have a lot of liveness. Valence is a sligly flat, left-skewed distribution. Acousticness shows a big spike close to the 0-10 range, meaning that most of the songs in the data base are not Acoustic.

It definately makes sense to normalize that data to ensure that there is some level of equivalence among features.

# 3. Principal Component Analysis

Now we can perform a principal component analysis of only 2 components.


```python
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
```


    
![png](output_13_0.png)
    



    
![png](output_13_1.png)
    


As we can see the fist principal component has higher coefficients for Energy, Valence, Speechiness, and Liveness; and the second principal component has higher coefficients for Dansability, and Acoustcness. Which could be interpreted as that the components are dividing the songs between those that have more lyrics and are more energitic, and those that have more melodies and are more for dancing. A 2x2 scatter plot shows that the observations are fairly evenly distributed across the two principal components.


```python
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
```

    Song with the highest value for the 1st principal component:
    Index                                      265
    Title                     Empire State Of Mind
    Artist                                   JAY-Z
    Top Genre                   east coast hip hop
    Year                                      2009
    Beats Per Minute (BPM)                     174
    Energy                                      96
    Danceability                                49
    Loudness (dB)                               -2
    Liveness                                    46
    Valence                                     81
    Length (Duration)                          277
    Acousticness                                 3
    Speechiness                                 39
    Popularity                                  77
    Name: 264, dtype: object
    
    Song with the lowest value for the 1st principal component:
    Index                                        1257
    Title                     Theme from Harry's Game
    Artist                                    Clannad
    Top Genre                                  celtic
    Year                                         1983
    Beats Per Minute (BPM)                        132
    Energy                                          7
    Danceability                                   19
    Loudness (dB)                                 -24
    Liveness                                       11
    Valence                                         4
    Length (Duration)                             148
    Acousticness                                   95
    Speechiness                                     4
    Popularity                                     38
    Name: 1256, dtype: object
    
    Song with the highest value for the 2nd principal component:
    Index                              1478
    Title                     Kingston Town
    Artist                             UB40
    Top Genre                 reggae fusion
    Year                               1989
    Beats Per Minute (BPM)              102
    Energy                               25
    Danceability                         96
    Loudness (dB)                       -13
    Liveness                              4
    Valence                              80
    Length (Duration)                   228
    Acousticness                         16
    Speechiness                           6
    Popularity                           49
    Name: 1477, dtype: object
    
    Song with the lowest value for the 2nd principal component:
    Index                                   95
    Title                       Limburg - Live
    Artist                         Rowwen Hèze
    Top Genre                 carnaval limburg
    Year                                  2008
    Beats Per Minute (BPM)                 157
    Energy                                  93
    Danceability                            23
    Loudness (dB)                           -5
    Liveness                                93
    Valence                                 28
    Length (Duration)                      403
    Acousticness                            29
    Speechiness                             17
    Popularity                              32
    Name: 94, dtype: object


As we can see the top song in the first principal component is "Empire State Of Mind" by Jay-Z, a song with high energy and a lot of lyrics. And the lowest rated song in that component is "Theme from Harry's Game" by Clannad, which is a celtic song. For the second component, the highest song is "Kingston Town" by UB40, and the lowest is "Limburg - Live" by Rowwen Hèze. In that sense we could see that the fist component is reflecting a little bit more the lyrics, and the second is reflecting the melodies.

# 4. Recommendation System

Let's first try to create the function calculating the distance over the 6 feautures.


```python
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
```

    Input Song: Without Me by Eminem
    
    Recommended Songs:
    Jump Around by House Of Pain
    I Will Survive - Single Version by Gloria Gaynor
    24K Magic by Bruno Mars
    Major Tom by Peter Schilling
    1999 by Prince
          Index                            Title           Artist     Top Genre  \
    1589   1590                      Jump Around    House Of Pain  gangster rap   
    1610   1611  I Will Survive - Single Version    Gloria Gaynor         disco   
    669     670                        24K Magic       Bruno Mars     dance pop   
    326     327                        Major Tom  Peter Schilling    german pop   
    1217   1218                             1999           Prince          funk   
    
          Year  Beats Per Minute (BPM)  Energy  Danceability  Loudness (dB)  \
    1589  1992                     107      71            85             -6   
    1610  1993                     116      62            80            -13   
    669   2016                     107      80            82             -4   
    326   2005                     164      54            78             -6   
    1217  1982                     119      73            87             -8   
    
          Liveness  Valence Length (Duration)  Acousticness  Speechiness  \
    1589        17       82               215             1            8   
    1610        32       64               199             2            5   
    669         15       63               226             3            8   
    326         22       68               274             3            6   
    1217         8       63               379            14            8   
    
          Popularity  
    1589          74  
    1610          60  
    669           78  
    326           48  
    1217          68  


Finaly, I creaded the recommendation function, using eucliden distance to find the closest 5 neighboors of the selected song. I tested it with "Without Me" by Eminem, and the recommendations were Jump Around by House Of Pain, I Will Survive - Single Version by Gloria Gaynor, 24K Magic by Bruno Mars, Major Tom by Peter Schilling, 1999 by Prince. I would say they are fairly good recommendations. I will now try to calculate the distance over only the two principal components.


```python
# Function to recommend songs using Euclidean distance on principal components
def recommend_songs(index_value, num_recommendations=5):
    # Find the row index for the given 'Index' column value
    song_index = df[df['Index'] == index_value].index[0]
    
    # Calculate the Euclidean distances between the selected song and all other songs using principal components
    distances = cdist(pca_df.iloc[song_index].values.reshape(1, -1), pca_df, metric='euclidean').flatten()
    
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
```

    Input Song: Without Me by Eminem
    
    Recommended Songs:
    Nutbush City Limits by Ike & Tina Turner
    Cecilia by Simon & Garfunkel
    Ain't Nobody by Chaka Khan
    Atemlos durch die Nacht by Helene Fischer
    Late in the Evening by Paul Simon
          Index                    Title             Artist           Top Genre  \
    397     398      Nutbush City Limits  Ike & Tina Turner  brill building pop   
    826     827                  Cecilia  Simon & Garfunkel        classic rock   
    1714   1715             Ain't Nobody         Chaka Khan           dance pop   
    733     734  Atemlos durch die Nacht     Helene Fischer          german pop   
    1185   1186      Late in the Evening         Paul Simon        classic rock   
    
          Year  Beats Per Minute (BPM)  Energy  Danceability  Loudness (dB)  \
    397   2008                      77      89            69             -6   
    826   1970                     103      88            76             -9   
    1714  1996                     104      88            80             -7   
    733   2018                     128      78            81             -4   
    1185  1980                     119      88            72             -9   
    
          Liveness  Valence Length (Duration)  Acousticness  Speechiness  \
    397          3       91               182             9            3   
    826         22       95               175            36            4   
    1714        14       82               281            19            4   
    733         19       82               220             6            5   
    1185         9       96               243            20            4   
    
          Popularity  
    397           56  
    826           73  
    1714          70  
    733           38  
    1185          58  



```python
Only using the principal component to estimate the distance did not work that well, as I did not know any of the recommended songs in this case. I guess it can proabily be because using only 2 components is not capturing enough complexity of the data to find a good recommendation.
```

# 5. Extra work

## 5.A. Recommendation Fuction including popularity

First I am going to run the recommendation system using the 6 features, and adding popularity.


```python
# Function to recommend songs using Euclidean distance and sort by popularity
def recommend_songs_with_popularity(index_value, num_recommendations=5):
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

# Test the recommendation engine with popularity
recommendations_with_popularity = recommend_songs_with_popularity(sample_song_index_value)
print(recommendations_with_popularity)
```

    Input Song: Without Me by Eminem
    
    Recommended Songs:
    24K Magic by Bruno Mars (Popularity: 78)
    Jump Around by House Of Pain (Popularity: 74)
    1999 by Prince (Popularity: 68)
    I Will Survive - Single Version by Gloria Gaynor (Popularity: 60)
    Major Tom by Peter Schilling (Popularity: 48)
          Index                            Title           Artist     Top Genre  \
    669     670                        24K Magic       Bruno Mars     dance pop   
    1589   1590                      Jump Around    House Of Pain  gangster rap   
    1217   1218                             1999           Prince          funk   
    1610   1611  I Will Survive - Single Version    Gloria Gaynor         disco   
    326     327                        Major Tom  Peter Schilling    german pop   
    
          Year  Beats Per Minute (BPM)  Energy  Danceability  Loudness (dB)  \
    669   2016                     107      80            82             -4   
    1589  1992                     107      71            85             -6   
    1217  1982                     119      73            87             -8   
    1610  1993                     116      62            80            -13   
    326   2005                     164      54            78             -6   
    
          Liveness  Valence Length (Duration)  Acousticness  Speechiness  \
    669         15       63               226             3            8   
    1589        17       82               215             1            8   
    1217         8       63               379            14            8   
    1610        32       64               199             2            5   
    326         22       68               274             3            6   
    
          Popularity  
    669           78  
    1589          74  
    1217          68  
    1610          60  
    326           48  


As we can see, I got the same results as before, but just in a different order. I will try to find the distance using the 2 first principal components and then adding popularity.


```python
# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Normalize the popularity feature
popularity_scaled = scaler.fit_transform(df[['Popularity']])

# Combine the principal components and the normalized popularity feature
combined_features = pd.DataFrame(principal_components, columns=['Principal Component 1', 'Principal Component 2'])
combined_features['Popularity'] = popularity_scaled

# Function to recommend songs using Euclidean distance on principal components and popularity
def recommend_songs(index_value, num_recommendations=5):
    # Find the row index for the given 'Index' column value
    song_index = df[df['Index'] == index_value].index[0]
    
    # Calculate the Euclidean distances between the selected song and all other songs using combined features
    distances = cdist(combined_features.iloc[song_index].values.reshape(1, -1), combined_features, metric='euclidean').flatten()
    
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
```

    Input Song: Without Me by Eminem
    
    Recommended Songs:
    Wannabe by Spice Girls
    I Wanna Dance with Somebody (Who Loves Me) by Whitney Houston
    All Star by Smash Mouth
    Get Lucky (feat. Pharrell Williams & Nile Rodgers) - Radio Edit by Daft Punk
    September by Earth, Wind & Fire
          Index                                              Title  \
    1706   1707                                            Wannabe   
    1411   1412         I Wanna Dance with Somebody (Who Loves Me)   
    1815   1816                                           All Star   
    539     540  Get Lucky (feat. Pharrell Williams & Nile Rodg...   
    730     731                                          September   
    
                      Artist         Top Genre  Year  Beats Per Minute (BPM)  \
    1706         Spice Girls         dance pop  1996                     110   
    1411     Whitney Houston         dance pop  1987                     119   
    1815         Smash Mouth  alternative rock  1999                     104   
    539            Daft Punk           electro  2013                     116   
    730   Earth, Wind & Fire             disco  2018                     126   
    
          Energy  Danceability  Loudness (dB)  Liveness  Valence  \
    1706      86            77             -6        16       89   
    1411      82            71             -9         9       87   
    1815      87            73             -6         9       78   
    539       81            79             -9        10       86   
    730       83            69             -7        25       98   
    
         Length (Duration)  Acousticness  Speechiness  Popularity  
    1706               173            10            3          80  
    1411               291            21            5          81  
    1815               200             4            3          80  
    539                248             4            4          77  
    730                215            17            3          82  


Now I feel this recommendation system is much better as I do like a lot all the recommended songs and feel that they are similar to the index song that I used. I guess if there was a third principal component it would be very similar to popularity, because it seems that it is capturing a lot of the signal needed to find a good recommendation.

## 5.B Principal Components Analysis from Scratch


```python
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
```

    [[ 1.66475637  1.6484432 ]
     [-1.33320148 -0.04190957]
     [-1.61114162  0.24698421]
     ...
     [-0.86739858  0.44367778]
     [ 1.4034789   0.95478631]
     [ 0.58090869  1.06383855]]


## 5.C Finding clusters


```python
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
```


    
![png](output_33_0.png)
    



    
![png](output_33_1.png)
    



```python
As we can see, we the optimal number of clusters is around 5, although with 3 clusters we have already categorized a lot of the information. 
```
