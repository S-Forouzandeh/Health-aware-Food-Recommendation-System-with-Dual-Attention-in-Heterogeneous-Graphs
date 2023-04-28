import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch


# Specify the directory path where the "Food" folder is located
folder_path = r"C:\Food"

# List of files to read
files_to_read = ['interactions_test.csv', 'interactions_train.csv', 'interactions_validation.csv', 'PP_recipes.csv',
                 'PP_users.csv', 'RAW_interactions.csv', 'RAW_recipes.csv']

# Create a dictionary to store user_id as key and total score and count as values
user_scores = {}

# Loop through the files and read their contents
for file in files_to_read:
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        # print(f"Reading file: {file}")a
        df = pd.read_csv(file_path)

        # Extract user_id and rating from 'RAW_interactions.csv' file
        if file == 'RAW_interactions.csv':
            user_id = df['user_id']
            rating = df['rating']

            # Iterate through user_id and rating columns to calculate total score and count for each user_id
            for i in range(len(user_id)):
                uid = user_id[i]
                r = rating[i]
                if uid not in user_scores:
                    user_scores[uid] = {'total_score': 0, 'count': 0}
                user_scores[uid]['total_score'] += r
                user_scores[uid]['count'] += 1

# Calculate the average score for each user_id
for uid, scores in user_scores.items():
    avg_score = scores['total_score'] / scores['count']
    print(f"User ID: {uid}, Average Score: {avg_score}")

# Extract ingredients and nutrition from 'RAW_recipes.csv' file
df = pd.read_csv(os.path.join(folder_path, 'RAW_recipes.csv'))
recipe_id = df['id']
ingredients = df['ingredients']
nutrition = df['nutrition']
# Define the health foods list
health_foods_list = ["Proteins", "Carbohydrates", "Sugars", "Sodium", "Fat", "Saturated fats", "Fibers"]


print(f"Recipe ID: {recipe_id}")
print(f"Ingredients: {ingredients}")
print(f"Nutrition: {nutrition}")

#-----------------Heterogeneous graph-------------------

# # Specify the directory path where the "Food" folder is located
# folder_path = r"C:\Food"
#
# # List of files to read
# files_to_read = ['interactions_test.csv', 'interactions_train.csv', 'interactions_validation.csv', 'PP_recipes.csv',
#                  'PP_users.csv', 'RAW_interactions.csv', 'RAW_recipes.csv']
#
# # Create a dictionary to store user_id as key and total score and count as values
# user_scores = {}
#
# # Create a directed multigraph
# G = nx.MultiDiGraph()
#
# # Loop through the files and read their contents
# for file in files_to_read:
#     file_path = os.path.join(folder_path, file)
#     if os.path.isfile(file_path):
#         # print(f"Reading file: {file}")
#         df = pd.read_csv(file_path)
#
#         # Extract user_id and rating from 'RAW_interactions.csv' file
#         if file == 'RAW_interactions.csv':
#             user_id = df['user_id']
#             recipe_id = df['recipe_id']
#             rating = df['rating']
#
#             # Iterate through user_id, recipe_id, and rating columns to add edges with weights to the graph
#             for i in range(len(user_id)):
#                 uid = user_id[i]
#                 rid = recipe_id[i]
#                 r = rating[i]
#
#                 # Add user_id and recipe_id as nodes
#                 G.add_node(uid, node_type='user')
#                 G.add_node(rid, node_type='recipe')
#
#                 # Add edge between user_id and recipe_id with weight as the rating
#                 G.add_edge(uid, rid, weight=r, edge_type='rating')
#
#         # Extract recipe_id, ingredients, and nutrition from 'RAW_recipes.csv' file
#         elif file == 'RAW_recipes.csv':
#             recipe_id = df['id']
#             ingredients = df['ingredients']
#             nutrition = df['nutrition']
#
#             # Iterate through recipe_id, ingredients, and nutrition columns to add edges to the graph
#             for i in range(len(recipe_id)):
#                 rid = recipe_id[i]
#                 ing = ingredients[i]
#                 nut = nutrition[i]
#
#                 # Add recipe_id, ingredients, and nutrition as nodes
#                 G.add_node(rid, node_type='recipe')
#                 G.add_node(ing, node_type='ingredient')
#                 G.add_node(nut, node_type='nutrition')
#
#                 # Add edges between recipe_id and ingredients/nutrition
#                 G.add_edge(rid, ing, edge_type='ingredient')
#                 G.add_edge(rid, nut, edge_type='nutrition')

# Draw the graph
# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
# nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color='gray')
# nx.draw_networkx_labels(G, pos, labels={}, font_size=8, font_color='black')
# plt.title('Heterogeneous Graph')
# plt.axis('off')
# plt.show()


# Specify the directory path where the "Food" folder is located
folder_path = r"C:\Food"

# List of files to read
files_to_read = ['interactions_test.csv', 'interactions_train.csv', 'interactions_validation.csv', 'PP_recipes.csv',
                 'PP_users.csv', 'RAW_interactions.csv', 'RAW_recipes.csv']

# Create a dictionary to store user_id as key and total score and count as values
user_scores = {}

# Create a directed multigraph
G = nx.MultiDiGraph()

# Loop through the files and read their contents
for file in files_to_read:
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        # print(f"Reading file: {file}")
        df = pd.read_csv(file_path)

        # Extract user_id and rating from 'RAW_interactions.csv' file
        if file == 'RAW_interactions.csv':
            user_id = df['user_id']
            recipe_id = df['recipe_id']
            rating = df['rating']

            # Iterate through user_id, recipe_id, and rating columns to add edges with weights to the graph
            for i in range(len(user_id)):
                uid = user_id[i]
                rid = recipe_id[i]
                r = rating[i]

                # Add user_id and recipe_id as nodes
                G.add_node(uid, node_type='user')
                G.add_node(rid, node_type='recipe')

                # Add edge between user_id and recipe_id with weight as the rating
                G.add_edge(uid, rid, weight=r, edge_type='rating')

        # Extract recipe_id, ingredients, and nutrition from 'RAW_recipes.csv' file
        elif file == 'RAW_recipes.csv':
            recipe_id = df['id']
            ingredients = df['ingredients']
            nutrition = df['nutrition']

            # Iterate through recipe_id, ingredients, and nutrition columns to add edges to the graph
            for i in range(len(recipe_id)):
                rid = recipe_id[i]
                ing = ingredients[i]
                nut = nutrition[i]

                # Add recipe_id, ingredients, and nutrition as nodes
                G.add_node(rid, node_type='recipe')
                G.add_node(ing, node_type='ingredient')
                G.add_node(nut, node_type='nutrition')

                # Add edges between recipe_id and ingredients/nutrition
                G.add_edge(rid, ing, edge_type='ingredient')
                G.add_edge(rid, nut, edge_type='nutrition')

    # Add edges between users and ingredients/nutrition based on rating score
    elif file == 'interactions_train.csv':
        user_id = df['user_id']
        recipe_id = df['recipe_id']
        rating = df['rating']
        avg_score = rating.mean()

        # Find recipes rated by user_id with score greater than or equal to avg_score
        rated_recipes = [node for node, attr in G.nodes(data=True) if
                         attr['node_type'] == 'recipe' and G.has_edge(uid, node)]
        rated_recipes_above_avg = [recipe for recipe in rated_recipes if
                                   G.get_edge_data(uid, recipe, 0)['weight'] >= avg_score]

        # Find ingredients/nutrition associated with the rated recipes and add edges between user and those nodes
        for recipe in rated_recipes_above_avg:
            ingredients = [node for node, attr in G.nodes(data=True) if
                           attr['node_type'] == 'ingredient' and G.has_edge(recipe, node) and attr[
                               'node_type'] == 'ingredient']
            nutrition = [node for node, attr in G.nodes(data=True) if
                         attr['node_type'] == 'nutrition' and G.has_edge(recipe, node) and attr[
                             'node_type'] == 'nutrition']
            for ing in ingredients:
                G.add_edge(uid, ing, edge_type='ingredient_rating')
            for nut in nutrition:
                G.add_edge(uid, nut, edge_type='nutrition_rating')

# Calculate average score for each user_id
for uid in user_scores:
    total_score = user_scores[uid][0]
    count = user_scores[uid][1]
    avg_score = total_score / count

    # Find recipes rated by user_id with score greater than or equal to avg_score
    rated_recipes = [node for node, attr in G.nodes(data=True) if
                     attr['node_type'] == 'recipe' and G.has_edge(uid, node)]
    rated_recipes_above_avg = [recipe for recipe in rated_recipes if G[uid][recipe][0] >= avg_score]

    print(f"User ID: {uid}")
    print(f"Average score: {avg_score}")
    print(f"Recipes rated by user: {rated_recipes}")
    print(f"Recipes rated above average: {rated_recipes_above_avg}")
    # Find ingredients and nutrition for recipes rated above average and add edges to the graph
    for recipe in rated_recipes_above_avg:
        recipe_idx = df[df['id'] == recipe].index[0]
        ingredients = df.loc[recipe_idx, 'ingredients']
        nutrition = df.loc[recipe_idx, 'nutrition']

        # Add edges between user_id and ingredients/nutrition
        for ingredient in ingredients:
            G.add_edge(uid, ingredient, edge_type='ingredient_rating')
        for nut in nutrition:
            G.add_edge(uid, nut, edge_type='nutrition_rating')

# -------------------------- Node Level Attention (NLA) based on the users popular foods -------------------------
class NodeLevelAttention:
    def __init__(self, graph, alpha=0.5):
        self.graph = graph
        self.alpha = alpha
        self.weights = None

    def _get_rated_recipes_above_avg(self, uid, avg_score):
        rated_recipes = [node for node, attr in self.graph.nodes(data=True) if
                         attr['node_type'] == 'recipe' and self.graph.has_edge(uid, node)]
        rated_recipes_above_avg = [recipe for recipe in rated_recipes if self.graph[uid][recipe][0] >= avg_score]
        return rated_recipes_above_avg

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _relu(self, x):
        return np.maximum(0, x)

    def _compute_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def _train_nla(self, uid, avg_score, y_true, epochs=100, learning_rate=0.001):
        rated_recipes_above_avg = self._get_rated_recipes_above_avg(uid, avg_score)
        num_recipes = len(rated_recipes_above_avg)

        # Initialize weights randomly
        self.weights = np.random.rand(num_recipes)

        for epoch in range(epochs):
            nla_scores = self.compute_nla_scores(uid, avg_score)
            y_pred = np.array([nla_scores[recipe] for recipe in rated_recipes_above_avg])

            # Update weights using gradient descent
            gradient = np.dot(y_true - y_pred, rated_recipes_above_avg)
            self.weights += learning_rate * gradient

    def compute_nla_scores(self, uid, avg_score):
        rated_recipes_above_avg = self._get_rated_recipes_above_avg(uid, avg_score)

        nla_scores = {}
        total_weight = 0.0
        for recipe in rated_recipes_above_avg:
            edge_weight = self.graph[uid][recipe][0]
            total_weight += edge_weight

        for recipe in rated_recipes_above_avg:
            edge_weight = self.graph[uid][recipe][0]
            nla_scores[recipe] = self._relu(self.alpha * edge_weight / total_weight)

        if len(nla_scores) > 0:
            nla_scores = self._softmax(list(nla_scores.values()))
            nla_scores_dict = {recipe: score for recipe, score in zip(rated_recipes_above_avg, nla_scores)}
        else:
            nla_scores_dict = {}

        return nla_scores_dict

# Instantiate NodeLevelAttention class
nla = NodeLevelAttention(G)

# Call compute_nla_scores method with uid and avg_score
uid = 123  # replace with desired uid
avg_score = 4.5  # replace with desired avg_score
nla_scores = nla.compute_nla_scores(uid, avg_score)

# Print the computed NLA scores
print("NLA scores for uid", uid, "with avg_score", avg_score, ":")
for recipe, score in nla_scores.items():
    print("Recipe:", recipe, "NLA Score:", score)

# Semantic Level Attention (SLA)------------------------------

class SemanticLevelAttention:
    def __init__(self, graph, health_foods_list, alpha=0.5):
        self.graph = graph
        self.alpha = alpha
        self.health_foods_list = health_foods_list
        self.weights = None

    def _get_rated_recipes_above_avg(self, uid, avg_score):
        rated_recipes = [node for node, attr in self.graph.nodes(data=True) if
                         attr['node_type'] == 'recipe' and self.graph.has_edge(uid, node)]

        rated_recipes_above_avg = []
        for recipe in rated_recipes:
            edge_weight = self.graph[uid][recipe][0]
            if edge_weight >= avg_score:
                ingredients = self.graph.nodes[recipe]['ingredients']
                health_food_count = 0
                for ingredient in ingredients:
                    if ingredient in self.health_foods_list:
                        health_food_count += 1
                if health_food_count >= 3:
                    rated_recipes_above_avg.append(recipe)

        return rated_recipes_above_avg

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _relu(self, x):
        return np.maximum(0, x)

    def edge_loss(W, S):
        """
        Computes the edge loss given the weight matrix W and the set of nodes S.
        """
        n = len(S)
        loss = 0
        for theta in range(1, n + 1):
            for u in range(1, n + 1):
                for F_H in range(1, n + 1):
                    for I in range(1, F_H + 1):
                        for N in range(1, F_H + 1):
                            Si = S[I]
                            Sj = S[N]
                            loss += -np.log(1 + np.exp(-W[Si, Sj]))
        return loss

    def _train_sla(self, uid, avg_score, y_true, epochs=100, learning_rate=0.001):
        rated_recipes_above_avg = self._get_rated_recipes_above_avg(uid, avg_score)
        num_recipes = len(rated_recipes_above_avg)

        # Initialize weights randomly
        self.weights = np.random.rand(num_recipes)

        for epoch in range(epochs):
            sla_scores = self.compute_sla_scores(uid, avg_score)
            y_pred = np.array([sla_scores[recipe] for recipe in rated_recipes_above_avg])

            # Update weights using gradient descent
            gradient = np.dot(y_true - y_pred, rated_recipes_above_avg)
            self.weights += learning_rate * gradient

            # Compute loss for monitoring
            loss = self._compute_loss(y_true, y_pred)
            print(f'Epoch: {epoch + 1}, Loss: {loss}')

    def compute_sla_scores(self, uid, avg_score):
        rated_recipes_above_avg = self._get_rated_recipes_above_avg(uid, avg_score)

        sla_scores = {}
        total_weight = 0.0
        for recipe in rated_recipes_above_avg:
            edge_weight = self.graph[uid][recipe][0]
            total_weight += edge_weight

        for recipe in rated_recipes_above_avg:
            edge_weight = self.graph[uid][recipe][0]
            sla_scores[recipe] = self._relu(self.alpha * edge_weight / total_weight)

        if len(sla_scores) > 0:
            sla_scores = self._softmax(list(sla_scores.values()))
            sla_scores_dict = {recipe: score for recipe, score in zip(rated_recipes_above_avg, sla_scores)}
        else:
            sla_scores_dict = {}

        return sla_scores_dict

# Create an instance of the SemanticLevelAttention class
sla = SemanticLevelAttention(G, health_foods_list)

# Call compute_sla_scores method to get SLA scores
uid = 000  # Example user ID
avg_score = 3  # Example average score
sla_scores = sla.compute_sla_scores(uid, avg_score)

# View the SLA scores
print("SLA Scores:")
for recipe, score in sla_scores.items():
    print(f"Recipe: {recipe}, SLA Score: {score}")

# ------ Health food Recommendation----------
class MyGraph:
    def __init__(self, graph, health_foods_list):
        self.graph = graph
        self.health_foods_list = health_foods_list

    def get_user_vectors(self, uid):
        # initialize empty vectors
        HF = []
        HI = []
        HN = []

        # get all rated recipes for the user
        rated_recipes = [node for node, attr in self.graph.nodes(data=True) if
                         attr['node_type'] == 'recipe' and self.graph.has_edge(uid, node)]

        for recipe in rated_recipes:
            # get ingredients for the recipe
            ingredients = self.graph.nodes[recipe]['ingredients']
            # check if any of the ingredients are in the health_foods_list
            for ingredient in ingredients:
                if ingredient in self.health_foods_list:
                    # add the ingredient to HF vector
                    if ingredient not in HF:
                        HF.append(ingredient)
                    # add the ingredient to HI vector if it has not been rated already
                    if ingredient not in HI and self.graph[uid][recipe][0] >= np.mean(
                            [self.graph[u][v][0] for u, v in self.graph.edges(uid)]):
                        HI.append(ingredient)
                    # add the nutrient information to HN vector if it has not been added already
                    if ingredient in HF and ingredient not in HN:
                        nutrient_info = self.graph.nodes[ingredient]['nutrient_info']
                        HN.append(nutrient_info)

        return HF, HI, HN

        # create an instance of the MyGraph class
        graph = MyGraph(G, health_foods_list)

        # set a valid user id
        uid = 123

        # call the get_user_vectors() method with the uid argument
        HF, HI, HN = graph.get_user_vectors(uid)

        # find the common elements between HF and HI
        common_HF_HI = set(HF).intersection(HI)
        print("Common elements between HF and HI:", common_HF_HI)

        # find the common elements between HF and HN
        common_HF_HN = set(HF).intersection(HN)
        print("Common elements between HF and HN:", common_HF_HN)

        # recommend elements from HF vector that are in both common_HF_HI and common_HF_HN
        recommendations = list(set(common_HF_HI).intersection(common_HF_HN))
        print("Recommendations:", recommendations)

# Final loss ------------------------
    class NodeLevelAttention:
        def _compute_loss(self, y_true, y_pred):
            return np.mean(np.square(y_true - y_pred))

    class SemanticLevelAttention:
        def __init__(self, weight_matrix):
            self.W = weight_matrix

        def _compute_loss(self, nodes):
            n = len(nodes)
            loss = 0
            for theta in range(1, n + 1):
                for u in range(1, n + 1):
                    for F_H in range(1, n + 1):
                        for I in range(1, F_H + 1):
                            for N in range(1, F_H + 1):
                                Si = nodes[I]
                                Sj = nodes[N]
                                loss += -np.log(1 + np.exp(-self.W[Si, Sj]))
            return loss


# ------------------------------ Evaluation

# Get unique user_ids and recipe_ids from the graph
user_ids = [node for node, attr in G.nodes(data=True) if attr['node_type'] == 'user']
recipe_ids = [node for node, attr in G.nodes(data=True) if attr['node_type'] == 'recipe']

# Create binary matrix with dimensions (num_users, num_recipes)
y_test = np.zeros((len(user_ids), len(recipe_ids)))

# Iterate through user_ids and rated recipes to fill in binary matrix
for i, uid in enumerate(user_ids):
    rated_recipes = [node for node, attr in G.nodes(data=True) if
                     attr['node_type'] == 'recipe' and G.has_edge(uid, node)]
    rated_recipes_above_avg = [recipe for recipe in rated_recipes if
                               G.get_edge_data(uid, recipe, 0)['weight'] >= avg_score]
    for j, rid in enumerate(recipe_ids):
        if rid in rated_recipes_above_avg:
            y_test[i, j] = 1


# Extract user_id, recipe_id, and rating from 'interactions_train.csv'
df_train = pd.read_csv(os.path.join(folder_path, 'interactions_train.csv'))
user_id_train = df_train['user_id']
recipe_id_train = df_train['recipe_id']
rating_train = df_train['rating']

# Split the ratings data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    pd.DataFrame({'user_id': user_id_train, 'recipe_id': recipe_id_train}),
    rating_train,
    test_size=0.2,  # adjust the test_size as needed
    random_state=42  # set the random state for reproducibility
)

# Create a binary training matrix with dimensions (num_users, num_recipes)
y_train_binary = np.zeros((len(user_ids), len(recipe_ids)))

# Iterate through training ratings to fill in binary matrix
for i, row in X_train.iterrows():
    uid = row['user_id']
    rid = row['recipe_id']
    j = recipe_ids.index(rid)  # find the index of the recipe in recipe_ids
    k = user_ids.index(uid)  # find the index of the user in user_ids
    y_train_binary[k, j] = 1 if y_train[i] >= avg_score else 0

#------------------------------
# initialize binary matrix with dimensions (num_users, num_recipes)
y_pred = np.zeros((len(user_ids), len(recipe_ids)))

# iterate through user_ids and rated recipes to fill in binary matrix
for i, uid in enumerate(user_ids):
    rated_recipes = [node for node, attr in G.nodes(data=True) if
                     attr['node_type'] == 'recipe' and G.has_edge(uid, node)]
    rated_recipes_above_avg = [recipe for recipe in rated_recipes if
                               G.get_edge_data(uid, recipe, 0)['weight'] >= avg_score]
    HF, HI, HN = G.get_user_vectors(uid)
    common_HF_HI = set(HF).intersection(HI)
    common_HF_HN = set(HF).intersection(HN)
    for j, rid in enumerate(recipe_ids):
        if rid in rated_recipes_above_avg and rid in common_HF_HI and rid in common_HF_HN:
            y_pred[i, j] = 1

# Set a seed for reproducibility
np.random.seed(123)

# Randomly select 80% of the ratings data for training
train_indices = np.random.choice(len(user_ids), size=int(0.8*len(user_ids)), replace=False)

# Create binary matrix with dimensions (num_users, num_recipes)
y_train = np.zeros((len(user_ids), len(recipe_ids)))

# Iterate through user_ids and rated recipes to fill in binary matrix
for i, uid in enumerate(user_ids):
    rated_recipes = [node for node, attr in G.nodes(data=True) if
                     attr['node_type'] == 'recipe' and G.has_edge(uid, node)]
    rated_recipes_above_avg = [recipe for recipe in rated_recipes if
                               G.get_edge_data(uid, recipe, 0)['weight'] >= avg_score]
    for j, rid in enumerate(recipe_ids):
        if rid in rated_recipes_above_avg:
            if i in train_indices:  # only update training set
                y_train[i, j] = 1
            else:  # set test set to 1
                y_test[i, j] = 1

# -----------------
def evaluate_model(model, X_test, y_test):
    y_pred = model(X_test)

    # Compute AUC
    auc_score = roc_auc_score(y_test, y_pred)

    # Compute AUC@k
    k_values = [5, 10, 15, 20, 30, 40, 50]
    auc_k_scores = []
    for k in k_values:
        auc_k_score = roc_auc_score(y_test[:,1], y_pred[:,1], max_fpr=1/k)
        auc_k_scores.append(auc_k_score)

    # Compute NDCG@k
    y_test_scores = y_test[:,1]
    y_pred_scores = y_pred[:,1]
    ndcg_k_scores = []
    for k in k_values:
        ndcg_k_score = ndcg_score([y_test_scores], [y_pred_scores], k=k)
        ndcg_k_scores.append(ndcg_k_score)

    # Compute Recall@k
    recall_k_scores = []
    recall_top_k_scores = []
    for k in k_values:
        y_pred_top_k = torch.argsort(y_pred, dim=0)[-k:]
        recall_k_score = recall_score(y_test.numpy()[:,1], y_pred.numpy()[:,1] > 0.5, pos_label=1, average='binary', labels=[1], sample_weight=None)
        recall_top_k_score = recall_score(y_test.numpy()[:,1], np.sum(y_pred_top_k.numpy(), axis=1) > 0.5, pos_label=1, average='binary', labels=[1], sample_weight=None)
        recall_k_scores.append(recall_k_score)
        recall_top_k_scores.append(recall_top_k_score)

    print(f"AUC: {auc_score}")
    for i, k in enumerate(k_values):
        print(f"AUC@{k}: {auc_k_scores[i]}")
    for i, k in enumerate(k_values):
        print(f"NDCG@{k}: {ndcg_k_scores[i]}")
    for i, k in enumerate(k_values):
        print(f"Recall@{k}: {recall_k_scores[i]}")
    for i, k in enumerate(k_values):
        print(f"Recall@{k}_top_k: {recall_top_k_scores[i]}")

    return auc_score, auc_k_scores, ndcg_k_scores, recall_k_scores, recall_top_k_scores

# ------------------------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Compute AUC
    auc_score = roc_auc_score(y_test, y_pred)

    # Compute AUC@k
    k_values = list(range(1, 21))
    auc_k_scores = []
    for k in k_values:
        auc_k_score = roc_auc_score(y_test[:, 1], y_pred[:, 1], max_fpr=1/k)
        auc_k_scores.append(auc_k_score)

    # Compute NDCG@k
    y_test_scores = y_test[:, 1]
    y_pred_scores = y_pred[:, 1]
    ndcg_k_scores = []
    for k in k_values:
        ndcg_k_score = ndcg_score([y_test_scores], [y_pred_scores], k=k)
        ndcg_k_scores.append(ndcg_k_score)

    # Compute Recall@k
    recall_k_scores = []
    recall_top_k_scores = []
    for k in k_values:
        y_pred_top_k = np.argsort(y_pred, axis=0)[-k:]
        recall_k_score = recall_score(y_test[:, 1], y_pred[:, 1] > 0.5, pos_label=1, average='binary', labels=[1], sample_weight=None)
        recall_top_k_score = recall_score(y_test[:, 1], np.sum(y_pred_top_k, axis=1) > 0.5, pos_label=1, average='binary', labels=[1], sample_weight=None)
        recall_k_scores.append(recall_k_score)
        recall_top_k_scores.append(recall_top_k_score)

    print(f"AUC: {auc_score}")
    for i, k in enumerate(k_values):
        print(f"AUC@{k}: {auc_k_scores[i]}")
    for i, k in enumerate(k_values):
        print(f"NDCG@{k}: {ndcg_k_scores[i]}")
    for i, k in enumerate(k_values):
        print(f"Recall@{k}: {recall_k_scores[i]}")
    for i, k in enumerate(k_values):
        print(f"Recall@{k}_top_k: {recall_top_k_scores[i]}")

    return auc_score, auc_k_scores, ndcg_k_scores, recall_k_scores, recall_top_k_scores

# ----------Clustering------------------------
def group_recipes_by_rating(nla, uid):
    recipe_clusters = {1: [], 2: [], 3: [], 4: []}

# Get all the rated recipes for the user
    rated_recipes = [node for node, attr in nla.graph.nodes(data=True) if
                     attr['node_type'] == 'recipe' and nla.graph.has_edge(uid, node)]

# Group the recipes into their respective clusters based on the rating given by the user
    for recipe in rated_recipes:
        rating = nla.graph[uid][recipe][0]
        if rating >= 0 and rating <= 5:
            recipe_clusters[1].append(recipe)
        elif rating >= 6 and rating <= 20:
            recipe_clusters[2].append(recipe)
        elif rating >= 21 and rating <= 50:
            recipe_clusters[3].append(recipe)
        elif rating > 50:
            recipe_clusters[4].append(recipe)

# Add an additional cluster for recipes rated by >50 user_ids
    for cluster_num in range(1, 5):
        recipe_clusters[cluster_num] = [recipe for recipe in recipe_clusters[cluster_num] if
                                        len(nla._get_rated_recipes_above_avg(recipe, 50)) < 50]

    recipe_clusters[5] = [recipe for recipe in rated_recipes if
                          len(nla._get_rated_recipes_above_avg(recipe, 50)) >= 50]
    return recipe_clusters

def get_avg_score_for_user(graph, uid):
    rated_recipes = [node for node, attr in graph.nodes(data=True) if attr['node_type'] == 'recipe' and graph.has_edge(uid, node)]
    total_score = sum(graph[uid][recipe][0] for recipe in rated_recipes)
    avg_score = total_score / len(rated_recipes) if len(rated_recipes) > 0 else 0
    return avg_score

def compute_auc_for_cluster(graph, cluster, nla):
    recipe_ids = []
    true_labels = []
    predicted_scores = []

    for uid, attr in graph.nodes(data=True):
        if attr['node_type'] == 'user':
            avg_score = get_avg_score_for_user(graph, uid)
            rated_recipes_above_avg = nla._get_rated_recipes_above_avg(uid, avg_score)

            for recipe in rated_recipes_above_avg:
                rating = graph[uid][recipe][0]
                if cluster == 1 and rating >= 0 and rating <= 5:
                    recipe_ids.append(recipe)
                    true_labels.append(1)
                    predicted_scores.append(nla.compute_nla_scores(uid, avg_score)[recipe])
                elif cluster == 2 and rating >= 6 and rating <= 20:
                    recipe_ids.append(recipe)
                    true_labels.append(1)
                    predicted_scores.append(nla.compute_nla_scores(uid, avg_score)[recipe])
                elif cluster == 3 and rating >= 21 and rating <= 50:
                    recipe_ids.append(recipe)
                    true_labels.append(1)
                    predicted_scores.append(nla.compute_nla_scores(uid, avg_score)[recipe])
                elif cluster == 4 and rating > 50:
                    recipe_ids.append(recipe)
                    true_labels.append(1)
                    predicted_scores.append(nla.compute_nla_scores(uid, avg_score)[recipe])

    auc_score = roc_auc_score(true_labels, predicted_scores)
    return auc_score

auc_cluster1 = compute_auc_for_cluster(G, 1, nla)
auc_cluster2 = compute_auc_for_cluster(G, 2, nla)
auc_cluster3 = compute_auc_for_cluster(G, 3, nla)
auc_cluster4 = compute_auc_for_cluster(G, 4, nla)

print("AUC for cluster 1: {:.4f}".format(auc_cluster1))
print("AUC for cluster 2: {:.4f}".format(auc_cluster2))
print("AUC for cluster 3: {:.4f}".format(auc_cluster3))
print("AUC for cluster 4: {:.4f}".format(auc_cluster4))
