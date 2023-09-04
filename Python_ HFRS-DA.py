import os
import pandas as pd
import networkx as nx
import math
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import torch_geometric.utils as utils
from dgl.nn.pytorch import GATConv
from torch_geometric.utils import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
import uuid
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.optim as optim
import dgl.function as fn
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score, ndcg_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import zipfile

folder_path = r"C:\Food"
files_to_read = ['Food_Dataset.zip']
file_path = r"C:\Food\Food_Dataset.zip"

# Read the file into a pandas DataFrame
df = pd.read_csv(file_path)

def process_data(folder_path, files_to_read):
    # Create a dictionary to store recipe_id as key and total score and count as values
    recipe_scores = {}

    # Loop through the files and read their contents
    for file in files_to_read:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # Read the CSV file
            if file == 'Food_Dataset.zip':
                df = pd.read_csv(file_path)
                user_id = df['user_id']
                recipe_id = df['recipe_id']
                rating = df['rating']

                # Iterate through user_id, recipe_id, and rating columns to calculate total score and count for each recipe_id
                for i in range(len(user_id)):
                    uid = user_id[i]
                    rid = recipe_id[i]
                    r = rating[i]
                    if rid not in recipe_scores:
                        recipe_scores[rid] = {'total_score': 0, 'count': 0}
                    recipe_scores[rid]['total_score'] += r
                    recipe_scores[rid]['count'] += 1

    # # Calculate the average score for each recipe_id
    # for rid, scores in recipe_scores.items():
    #     avg_score = scores['total_score'] / scores['count']
    #     print(f"Recipe ID: {rid}, Average Score: {avg_score}")

    # Extract ingredients and nutrition from 'Food_Dataset.zip' file
    df = pd.read_csv(os.path.join(folder_path, 'Food_Dataset.zip'))
    recipe_id = df['recipe_id']
    ingredients = df['ingredients']
    nutrition = df['nutrition']

    # Print the first 10 user_ids along with their information
    for i in range(min(3, len(df))):
        uid = df.loc[i, 'user_id']
        rid = df.loc[i, 'recipe_id']
        rat = df.loc[i, 'rating']
        ing = df.loc[i, 'ingredients']
        nut = df.loc[i, 'nutrition']

        print(f"User ID: u{uid}")
        print(f"Recipe ID: r{rid}")
        print(f"Rating: {rat}")
        print(f"Ingredients: {ing}")
        print(f"Nutrition: {nut}")
        print()

def Load_Into_Graph(df):
    """Given a data frame with columns 'user_id', 'recipe_id',
    'ingredients', and 'nutrition', construct a multigraph with the
    following schema:

    Nodes:
    * user: identified with user_id
    * recipe: identified with recipe_id
    * ingredients: identified with ingredient string
    * nutrient: one of nutrients below

    Edges:
    * user -> recipe, if user rated recipe, with the rating as the weight
    * recipe -> ingredients, if recipe contains that ingredient
    * recipe -> nutrient, if recipe contains that nutrient, with the amount as the weight

    Note: Ingredient and nutrient lists are included in the data frame
        as Python-like lists, e.g., "['salt', 'wheat flour', 'rice']"
        for ingredients and [1,.5,0] for nutrients. They are therefore
        decoded.

    """

    print("Loading data into a graph...")
    
    # Create an empty graph
    G = nx.Graph()

    nutrients = ["Proteins", "Carbohydrates", "Sugars",
                "Sodium", "Fat", "Saturated_fats", "Fibers"]
    G.add_nodes_from(nutrients, node_type='nutrition')

    # Iterate through the data and populate the graph
    for uid, rid, r, ing, nut in df[['user_id', 'recipe_id', 'rating', 'ingredients', 'nutrition']].itertuples(False, None):
        # Add user_id, recipe_id
        G.add_node(f"u{uid}", node_type='user')
        G.add_node(f"r{rid}", node_type='recipe')

        # Add edges between user_id and recipe_id
        G.add_edge(f"u{uid}", f"r{rid}", weight=r, edge_type='rating')

        # Add new ingredients as nodes
        if type(ing) is str:
            # Remove brackets and single quotes
            ing = eval(ing)
            G.add_nodes_from(ing, node_type='ingredients')
            # Add edges between recipe_id and ingredients
            for i in ing: 
                G.add_edge(f"r{rid}", i, edge_type='ingredient')

        # Add edges between recipe_id and nutrients
        if type(nut) is str:
            nuts = eval(nut)
            for j, nut in enumerate(nutrients):
                if nuts[j] > 0:
                    G.add_edge(f"r{rid}", nut, weight=nuts[j], edge_type='nutrition')

    print("Finished; resulting graph:")
    print(G)
    return G

def Heterogeneous_Graph(df):
    # Populate the heterogeneous graph
    G = Load_Into_Graph(df)

    # Define the meta-paths
    meta_paths = [
        ['user_id', 'recipe_id', 'nutrition', 'ingredients'],
        ['user_id', 'recipe_id'],
        ['user_id', 'recipe_id', 'ingredients', 'nutrition'],
        ['recipe_id', 'nutrition', 'ingredients'],
        ['recipe_id', 'ingredients', 'nutrition']
    ]

    # Print the edges and their attributes for each meta-path
    for meta_path in meta_paths:
        print("Meta-Path:", " -> ".join(meta_path))
        
        paths = []

        # Check if the meta-path starts with 'user_id' and ends with 'ingredients'
        if meta_path[0] == 'user_id' and meta_path[-1] == 'ingredients':
            for uid in G.nodes():
                if G.nodes[uid]['node_type'] == 'user':
                    for rid in G.neighbors(uid):
                        if G.nodes[rid]['node_type'] == 'recipe':
                            for ing in G.neighbors(rid):
                                if G.nodes[ing]['node_type'] == 'ingredients':
                                    paths.append([f"{uid}", f"{rid}", ing])

        # Check if the meta-path starts with 'user_id' and ends with 'nutrition'
        elif meta_path[0] == 'user_id' and meta_path[-1] == 'nutrition':
            for uid in G.nodes():
                if G.nodes[uid]['node_type'] == 'user':
                    for rid in G.neighbors(uid):
                        if G.nodes[rid]['node_type'] == 'recipe':
                            for nut in G.neighbors(rid):
                                if G.nodes[nut]['node_type'] == 'nutrition':
                                    for ing in G.neighbors(rid):
                                        if G.nodes[ing]['node_type'] == 'ingredients':
                                            paths.append([f"{uid}", f"{rid}", nut, ing])

        
        # Print only the first 5 paths for each meta-path
        for i, path in enumerate(paths[:5]):
            print("Path:", path)
            for j in range(len(path) - 1):
                source = path[j]
                target = path[j + 1]
                edges = G.get_edge_data(source, target)
                if edges is not None:
                    for key, data in edges.items():
                        print("Source:", source)
                        print("Target:", target)
                        print("Edge Data:", data)  # Print all edge data
                else:
                    print("No edges between", source, "and", target)
            print()

# Define the NLA class
class NLA(nn.Module):
    def __init__(self, num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths):
        super(NLA, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.nutrition_embedding = nn.Embedding(num_nutrition, embedding_dim)

        # Convert the paths to tensors
        self.paths = paths.clone().detach() if paths is not None else None

    def forward(self, uid, rid, ing, nut):
        user_emb = self.user_embedding(uid)
        recipe_emb = self.recipe_embedding(rid)
        ingredient_emb = self.ingredient_embedding(ing)
        nutrition_emb = self.nutrition_embedding(nut)

        if self.paths is not None:
            path_scores = torch.zeros(uid.size(0), len(self.paths))
            for i, path in enumerate(self.paths):
                path = torch.tensor(path).clone().detach()
                matching_uid = torch.where(uid == path[0])[0]
                matching_rid = torch.where(rid == path[1])[0]
                matching_ing = torch.where(ing == path[2])[0]
                matching_nut = torch.where(nut == path[3])[0]  # Fix this line
                
                # Check if there are any matching indices
                if matching_uid.size(0) > 0 and matching_rid.size(0) > 0 and matching_ing.size(0) > 0 and matching_nut.size(0) > 0:
                    matching_count = min(matching_uid.size(0), matching_rid.size(0), matching_ing.size(0), matching_nut.size(0))
                    matching_indices = torch.stack((matching_uid[:matching_count], matching_rid[:matching_count], matching_ing[:matching_count], matching_nut[:matching_count]))
                    path_scores[matching_indices] += 1
                    
            # Node-Level Attention
            k = 3  # Number of iterations
            node_emb_theta = torch.zeros(user_emb.size(0), user_emb.size(1))
            for i in range(k):
                attention_scores = F.leaky_relu(node_emb_theta, negative_slope=0.01)
                attention_scores = F.softmax(attention_scores, dim=1)
                weighted_attention = attention_scores.unsqueeze(2) * user_emb.unsqueeze(1)

            aggregated_attention = torch.sum(weighted_attention, dim=1)

            # Determine the maximum size along dimension 0
            max_size = max(user_emb.size(0), user_emb.size(0), aggregated_attention.size(0))

            # Pad tensors to match the maximum size along dimension 0
            user_emb = F.pad(user_emb, (0, 0, 0, max_size - user_emb.size(0)))
            recipe_emb = F.pad(recipe_emb, (0, 0, 0, max_size - recipe_emb.size(0)))
            aggregated_attention = F.pad(aggregated_attention, (0, 0, 0, max_size - aggregated_attention.size(0)))

            # Concatenate and return the final embedding
            node_embeddings = torch.cat((user_emb, recipe_emb, aggregated_attention), dim=1)
        else:
            # Concatenate the embeddings without attention
            node_embeddings = torch.cat((user_emb, recipe_emb, ingredient_emb, nutrition_emb), dim=1)

        return node_embeddings

    def train_nla(self, df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder, num_epochs=100):
        criterion_nla = nn.MSELoss()
        optimizer_nla = optim.Adam(self.parameters(), lr=0.01)

        dataset = HeterogeneousDataset(df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            running_loss_nla = 0.0
            for uid, rid, ing, nut, label in data_loader:
                optimizer_nla.zero_grad()

                # Forward pass
                embeddings = self(uid, rid, ing, nut)
                label = label.unsqueeze(1).float()

                # Calculate the loss
                loss_nla = criterion_nla(embeddings, label)
                running_loss_nla += loss_nla.item()

                # Backward pass and optimization
                loss_nla.backward()
                optimizer_nla.step()

            avg_loss_nla = running_loss_nla / len(data_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, NLA Loss: {avg_loss_nla:.4f}")
        
        # Return the final NLA loss value
        return avg_loss_nla

    def get_embeddings(self, uid, rid, ing, nut):
        # Forward pass to get embeddings
        with torch.no_grad():
            embeddings = self(uid, rid, ing, nut)
        return embeddings
    
# Define the dataset class
class HeterogeneousDataset(Dataset):
    def __init__(self, df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder):
        self.uids = user_encoder.transform(df['user_id'])
        self.rids = recipe_encoder.transform(df['recipe_id'])
        self.ings = ingredient_encoder.transform(df['ingredients'])
        self.nuts = nutrition_encoder.transform(df['nutrition'])
        self.labels = df['rating'].astype(float).values

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        rid = self.rids[idx]
        ing = self.ings[idx]
        nut = self.nuts[idx]
        label = self.labels[idx]
        return uid, rid, ing, nut, label

def find_paths_users_interests(df):
    # Populate the heterogeneous graph
    G = Load_Into_Graph(df)

    # Calculate the average rating for each recipe_id and create a new column 'avg_rating'
    df['avg_rating'] = df.groupby('recipe_id')['rating'].mean()

    # Print the meta-path
    meta_path = ['user_id', 'recipe_id', 'ingredient', 'nutrition']
    print("Meta-Path:", " -> ".join(meta_path))

    paths = []
    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
            for rid in user_rated_recipes:
                # Check if there are matching rows in df before accessing by index
                matching_rows = df[df['recipe_id'] == rid]
                if not matching_rows.empty:
                    if matching_rows['rating'].iloc[0] >= matching_rows['avg_rating'].iloc[0]:  # Use 'avg_rating' from matching_rows
                        ingredient_node = []
                        nutrition_node = []

                        for node in G.neighbors(rid):
                            if G.nodes[node]['node_type'] == 'ingredients':
                                ingredient_node.append(node)
                            elif G.nodes[node]['node_type'] == 'nutrition':
                                nutrition_node.append(node)

                        for ing in ingredient_node:
                            for nut in nutrition_node:
                                paths.append([uid, rid, ing, nut])

    # Encode the paths using label encoders
    user_encoder = LabelEncoder()
    recipe_encoder = LabelEncoder()
    ingredient_encoder = LabelEncoder()
    nutrition_encoder = LabelEncoder()
    user_encoder.fit([path[0] for path in paths])
    recipe_encoder.fit([path[1] for path in paths])
    ingredient_encoder.fit([path[2] for path in paths])
    nutrition_encoder.fit([path[3] for path in paths])

    encoded_paths = [[user_encoder.transform([path[0]])[0], recipe_encoder.transform([path[1]])[0], ingredient_encoder.transform([path[2]])[0], nutrition_encoder.transform([path[3]])[0]] for path in paths]

    # Convert paths to tensors
    paths_tensor = torch.tensor(encoded_paths, dtype=torch.long).clone().detach()

    # Print the first 5 filtered paths
    for i, (path, encoded_path) in enumerate(zip(paths, encoded_paths)):
        print("Original Path:", path)
        print("Encoded Path:", encoded_path)
        if i == 5:
            break

    return paths_tensor, meta_path

# Define the SLA class
class SLA(nn.Module):
    def __init__(self, num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths, is_healthy=False):
        super(SLA, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.nutrition_embedding = nn.Embedding(num_nutrition, embedding_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),  # Output size matches embedding_dim
            nn.LeakyReLU(negative_slope=0.01),  
            nn.Softmax(dim=1)
        )
        self.is_healthy = is_healthy  # New parameter
        self.paths = paths.clone().detach() if paths is not None else None

    def forward(self, uid, rid, ing, nut, is_healthy=None):
        if is_healthy is None:
            is_healthy = self.is_healthy
        else:
            is_healthy = F.leaky_relu(is_healthy, negative_slope=0.01)

        user_emb = self.user_embedding(uid)
        recipe_emb = self.recipe_embedding(rid)
        ingredient_emb = self.ingredient_embedding(ing)
        nutrition_emb = self.nutrition_embedding(nut)

        # Determine the maximum size along dimension 0
        max_size = max(user_emb.size(0), recipe_emb.size(0), ingredient_emb.size(0), nutrition_emb.size(0))

        # Pad tensors to match the maximum size along dimension 0
        user_emb = F.pad(user_emb, (0, 0, 0, max_size - user_emb.size(0)))
        recipe_emb = F.pad(recipe_emb, (0, 0, 0, max_size - recipe_emb.size(0)))
        ingredient_emb = F.pad(ingredient_emb, (0, 0, 0, max_size - ingredient_emb.size(0)))
        nutrition_emb = F.pad(nutrition_emb, (0, 0, 0, max_size - nutrition_emb.size(0)))

        # Concatenate and return the final embedding
        node_embeddings = torch.cat((user_emb, recipe_emb, ingredient_emb, nutrition_emb), dim=1)

        return node_embeddings

    def edge_loss(self, h_sla):
        loss = -torch.log(1 / (1 + torch.exp(h_sla)))
        return loss.mean()

    def train_sla(self, uid_tensor, rid_tensor, ing_tensor, nut_tensor, num_epochs_sla=100):
        optimizer_sla = optim.Adam(self.parameters(), lr=0.01)

        for epoch_sla in range(num_epochs_sla):
            optimizer_sla.zero_grad()

            # Forward pass
            embeddings_for_healthy_foods = self(uid_tensor, rid_tensor, ing_tensor, nut_tensor)

            # Calculate the loss using the edge_loss function
            loss_sla = self.edge_loss(embeddings_for_healthy_foods)
            loss_sla.backward()
            optimizer_sla.step()

            # Print the loss for SLA
            print(f"Epoch SLA {epoch_sla + 1}/{num_epochs_sla}, SLA Loss: {loss_sla.item():.4f}")

        # Print the aggregated ingredient embeddings from SLA (for healthy recipes)
        print("Embeddings Vectors (SLA) based Healthy recipes:")
        print(embeddings_for_healthy_foods)

# Define the is_healthy function
def is_healthy(food_data):
    fibres = food_data[0]
    fat = food_data[1]
    sugar = food_data[2]
    sodium = food_data[3]
    protein = food_data[4]
    saturated_fat = food_data[5]
    carbohydrates = food_data[6]
    
    conditions_met = 0
    
    if fibres > 10:
        conditions_met += 1
    if 15 <= fat <= 30:
        conditions_met += 1
    if sugar < 10:
        conditions_met += 1
    if sodium < 5:
        conditions_met += 1
    if 10 <= protein <= 15:
        conditions_met += 1
    if saturated_fat < 10:
        conditions_met += 1
    if 55 <= carbohydrates <= 75:
        conditions_met += 1
    
    return conditions_met >= 3

def find_healthy_foods(df):
    # Populate the heterogeneous graph
    G = Load_Into_Graph(df)

    # Calculate the average rating for each recipe_id and create a new column 'avg_rating'
    df['avg_rating'] = df.groupby('recipe_id')['rating'].mean()

    paths = []
    healthy_foods = set()  # Store healthy recipes here

    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
            for rid in user_rated_recipes:
                # Check if there are matching rows in df before accessing by index
                matching_rows = df[df['recipe_id'] == rid]
                if not matching_rows.empty:
                    if matching_rows['rating'].iloc[0] >= matching_rows['avg_rating'].iloc[0]:
                        nutrition_health = [int(token) for token in eval(matching_rows['nutrition'].iloc[0]) if token.strip().isdigit()]
                        is_healthy_food = is_healthy(nutrition_health)
                        ingredient_node = []
                        nutrition_node = []
                        for node in G.neighbors(rid):
                            if G.nodes[node]['node_type'] == 'ingredients':
                                ingredient_node.append(node)
                            elif G.nodes[node]['node_type'] == 'nutrition':
                                nutrition_node.append(node)
                        for ing in ingredient_node:
                            for nut in nutrition_node:
                                paths.append([uid, rid, ing, nut])
                        if is_healthy_food:
                            healthy_foods.add(rid)  # Add the recipe to healthy foods

    # Encode the paths using label encoders
    recipe_encoder = LabelEncoder()
    recipe_encoder.fit(list(healthy_foods))
    encoded_paths = [[path[1]] for path in paths if path[1] in healthy_foods]

    # Convert paths to tensors
    paths_tensor = torch.tensor(encoded_paths, dtype=torch.long)
    
    return paths_tensor

def rate_healthy_recipes_for_user(user_id, df):
    # Filter the data for the specified user_id
    user_data = df[df['user_id'] == user_id]

    # Get the healthy recipes for the user
    user_healthy_recipes = set()
    for rid in user_data['recipe_id'].unique():
        avg_rating = user_data[user_data['recipe_id'] == rid]['avg_rating'].iloc[0]
        rating = user_data[user_data['recipe_id'] == rid]['rating'].iloc[0]
        if rating >= avg_rating:
            nutrition_health = eval(user_data[user_data['recipe_id'] == rid]['nutrition'].iloc[0])
            if is_healthy(nutrition_health):
                user_healthy_recipes.add(rid)

    return user_healthy_recipes

def recommend_users_for_healthy_recipes(df, embeddings_for_healthy_foods):
    recommendations = {}
    index_to_user_id = {}

    # Inverse mapping to get user IDs from indices
    for i, user_id in enumerate(df['user_id'].unique()):
        index_to_user_id[i] = user_id

    # Calculate cosine similarities between user embeddings
    similarities = cosine_similarity(embeddings_for_healthy_foods)  

    for i, user_embedding in enumerate(embeddings_for_healthy_foods):
        user_id = index_to_user_id[i]

        recommendations[user_id] = {
            'most_similar_user_id': None,
            'similar_users': [],
            'users_with_shared_ingredients': [],
            'users_with_shared_nutrition': [],
            'user_healthy_recipes': []
        }

        # Sort users by similarity (excluding the current user)
        similar_users_indices = similarities[i].argsort()[::-1]  # Sort in descending order
        most_similar_index = similar_users_indices[0]

        recommendations[user_id]['most_similar_user_id'] = index_to_user_id[most_similar_index]

        # Find users who share ingredients and nutrition based on embeddings
        for j, similarity_score in enumerate(similarities[i]):
            if i != j:
                common_ingredients = set(eval(df[df['user_id'] == user_id]['ingredients'].iloc[0])) & \
                                    set(eval(df[df['user_id'] == index_to_user_id[j]]['ingredients'].iloc[0]))
                
                if common_ingredients:
                    recommendations[user_id]['users_with_shared_ingredients'].append(index_to_user_id[j])

        # Find healthy recipes for the user
        user_healthy_recipes = rate_healthy_recipes_for_user(user_id, df)
        recommendations[user_id]['user_healthy_recipes'] = user_healthy_recipes

    return recommendations

def evaluate_recommendations(recommendations, ground_truth_ratings, validation_size=0.1, test_size=0.2):
    true_ratings = []
    recommended_ratings = []

    for user_id, recommended_user_ids in recommendations.items():
        if user_id in ground_truth_ratings:
            true_rating = float(ground_truth_ratings[user_id]['rating'])
            true_ratings.append(true_rating)

            for recommended_user_id in recommended_user_ids:
                if recommended_user_id in ground_truth_ratings:
                    recommended_rating = float(ground_truth_ratings[recommended_user_id]['rating'])
                    recommended_ratings.append(recommended_rating)

    if len(true_ratings) == 0 or len(recommended_ratings) == 0:
        print("Insufficient data for evaluation.")
        return None

    # Randomly shuffle the data
    combined_data = list(zip(true_ratings, recommended_ratings))
    random.shuffle(combined_data)
    true_ratings, recommended_ratings = zip(*combined_data)

    # Calculate the indices to split the data into test, validation, and training sets
    num_samples = len(true_ratings)
    validation_split_index = int(num_samples * (1 - validation_size))
    test_split_index = int(num_samples * (1 - test_size))

    # Split the data into training, validation, and test sets
    true_ratings_train = true_ratings[:validation_split_index]
    recommended_ratings_train = recommended_ratings[:validation_split_index]

    true_ratings_validation = true_ratings[validation_split_index:test_split_index]
    recommended_ratings_validation = recommended_ratings[validation_split_index:test_split_index]

    true_ratings_test = true_ratings[test_split_index:]
    recommended_ratings_test = recommended_ratings[test_split_index:]

    if len(true_ratings_validation) > 0 and len(recommended_ratings_validation) > 0:
        auc_score_validation = roc_auc_score(true_ratings_validation, recommended_ratings_validation)
        ndcg_score_validation = ndcg_score([true_ratings_validation], [recommended_ratings_validation])
        recall_score_validation = np.mean(np.equal(true_ratings_validation, recommended_ratings_validation))

        print("Validation Evaluation Scores:")
        print("AUC Score:", auc_score_validation)
        print("NDCG Score:", ndcg_score_validation)
        print("Recall Score:", recall_score_validation)

    if len(true_ratings_test) > 0 and len(recommended_ratings_test) > 0:
        auc_score_test = roc_auc_score(true_ratings_test, recommended_ratings_test)
        ndcg_score_test = ndcg_score([true_ratings_test], [recommended_ratings_test])
        recall_score_test = np.mean(np.equal(true_ratings_test, recommended_ratings_test))

        print("Testing Evaluation Scores:")
        print("AUC Score:", auc_score_test)
        print("NDCG Score:", ndcg_score_test)
        print("Recall Score:", recall_score_test)

    return auc_score_validation, ndcg_score_validation, recall_score_validation, auc_score_test, ndcg_score_test, recall_score_test

def main():

    # Call the process_data function
    process_data(folder_path, files_to_read)

    # Call the Heterogeneous_Graph function
    Heterogeneous_Graph(df)

   # Call the find_paths_users_interests function
    paths_tensor, meta_path = find_paths_users_interests(df)

    # Print the filtered meta-path
    print("Filtered Meta-Path:", " -> ".join(meta_path))

    # Get the unique node counts
    num_users = len(df['user_id'].unique())
    num_recipes = len(df['recipe_id'].unique())
    num_ingredients = len(df['ingredients'].unique())
    num_nutrition = len(df['nutrition'].unique())

    # Initialize the label encoders and fit them with the data
    user_encoder = LabelEncoder()
    recipe_encoder = LabelEncoder()
    ingredient_encoder = LabelEncoder()
    nutrition_encoder= LabelEncoder()
    user_encoder.fit(df['user_id'])
    recipe_encoder.fit(df['recipe_id'])
    ingredient_encoder.fit(df['ingredients'])
    nutrition_encoder.fit(df['nutrition'])
    
    # Initialize the NLA model
    embedding_dim = 64
    # Create an instance of the NLA class
    nla_model = NLA(num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths_tensor)

    # Train the NLA model
    nla_loss = nla_model.train_nla(df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder, num_epochs=100)

    # Get and print the embeddings
    uid_tensor = torch.LongTensor(list(range(num_users)))
    rid_tensor = torch.LongTensor(list(range(num_recipes)))
    ing_tensor = torch.LongTensor(list(range(num_ingredients)))
    nut_tensor = torch.LongTensor(list(range(num_nutrition)))
    embeddings_nla = nla_model.get_embeddings(uid_tensor, rid_tensor, ing_tensor, nut_tensor)

    print("Embedding Vectors (NLA):")
    print(embeddings_nla)
        
    # Create an SLA instance for healthy foods
    sla_for_healthy_foods = SLA(num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths_tensor, is_healthy=True)

    # Train the SLA model for healthy foods
    sla_for_healthy_foods.train_sla(uid_tensor, rid_tensor, ing_tensor, nut_tensor, num_epochs_sla=100)

    # Extract the embeddings for healthy foods after training
    embeddings_for_healthy_foods = sla_for_healthy_foods(uid_tensor, rid_tensor, ing_tensor, nut_tensor)

    # Detach the tensor before converting it to a NumPy array
    embeddings_for_healthy_foods = embeddings_for_healthy_foods.detach().numpy()

    # Compute cosine similarities
    similarities = cosine_similarity(embeddings_for_healthy_foods)

    recommendations = recommend_users_for_healthy_recipes(df, embeddings_for_healthy_foods)

    # Iterate through the recommendations for each user
    for user_id, user_recommendations in recommendations.items():
        print(f"User {user_id}'s Recommendations:")

        # Get the most similar user
        most_similar_user_id = user_recommendations['most_similar_user_id']
        print(f"Most Similar User: {most_similar_user_id}")

        # Get the top 5 users with shared ingredients
        shared_ingredients_users = user_recommendations['users_with_shared_ingredients'][:5]
        print("Users with Shared Ingredients:")
        for shared_user in shared_ingredients_users:
            print(shared_user)

    # Read the ground truth ratings into a dictionary
    ground_truth_ratings = {}
    for file in files_to_read:
        if file == 'Food_Dataset.zip':
            interactions_df = pd.read_csv(os.path.join(folder_path, file), dtype=str)
            for index, row in interactions_df.iterrows():
                user_id = row['user_id']
                rating = row['rating']
                ground_truth_ratings[user_id] = {'rating': rating}

    # Call the evaluate_recommendations function
    validation_size = 0.1  # Set the proportion of the training set for validation
    test_size = 0.2  # Set the proportion of the data for testing
    result = evaluate_recommendations(recommendations, ground_truth_ratings, validation_size, test_size)

if __name__ == '__main__':
    main()