import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding


model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


def step_one():
    """
    Preprocess the WebQSP dataset to create node and edge CSV files for each graph.

    This function performs the following steps:
    1. Loads the WebQSP dataset and concatenates the train, validation, and test splits into a single dataset.
    2. Creates directories to store node and edge files if they do not exist.
    3. Iterates over each graph in the dataset and extracts nodes and edges.
    4. Saves the extracted nodes and edges into separate CSV files for each graph.
    5. Each graph contains also a question

    The resulting node and edge files are used in the next preprocessing step to create graph embeddings.
    """
    
    # Load the WebQSP dataset and concatenate the train, validation, and test splits
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    # Create directories to store nodes and edges if they don't exist
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    # Iterate over each graph in the dataset
    for i in tqdm(range(len(dataset))):
        nodes = {}  # Dictionary to store nodes and their IDs
        edges = []  # List to store edges

        # Iterate over each triple (head, relation, tail) in the graph
        for tri in dataset[i]['graph']:
            h, r, t = tri  # Unpack the triple into head, relation, and tail
            h = h.lower()  # Convert head to lowercase
            t = t.lower()  # Convert tail to lowercase

            # Add head to nodes dictionary if not already present
            if h not in nodes:
                nodes[h] = len(nodes)
            
            # Add tail to nodes dictionary if not already present
            if t not in nodes:
                nodes[t] = len(nodes)

            # Append the edge (with source, relation, and destination) to edges list
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})

        # Convert nodes dictionary to a DataFrame
        nodes_df = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        
        # Convert edges list to a DataFrame
        edges_df = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        # Save nodes DataFrame to a CSV file
        nodes_df.to_csv(f'{path_nodes}/{i}.csv', index=False)
        
        # Save edges DataFrame to a CSV file
        edges_df.to_csv(f'{path_edges}/{i}.csv', index=False)



def generate_split():
    
    dataset = load_dataset("rmanluo/RoG-webqsp")

    train_indices = np.arange(len(dataset['train']))
    val_indices = np.arange(len(dataset['validation'])) + len(dataset['train'])
    test_indices = np.arange(len(dataset['test'])) + len(dataset['train']) + len(dataset['validation'])

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))


def step_two():
    """
    Encode questions and graph nodes/edges into embeddings and create PyTorch Geometric (PyG) graph data objects.

    This function performs the following steps:
    1. Loads the WebQSP dataset and concatenates the train, validation, and test splits into a single dataset.
    2. Extracts questions from the dataset and encodes them into embeddings using a pre-trained model.
    3. Loads node and edge data files generated in step_one for each graph.
    4. Encodes node attributes and edge attributes into embeddings using the same pre-trained model.
    5. Constructs PyTorch Geometric (PyG) Data objects for each graph with node embeddings, edge index, and edge attributes.
    6. Saves the encoded questions and PyG Data objects for further processing in the pipeline.

    This function relies on the node and edge files created by step_one to generate graph embeddings.
    """

    # Load the WebQSP dataset and concatenate the train, validation, and test splits
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    
    # Extract questions from the dataset
    questions = [i['question'] for i in dataset]

    # Load pre-trained model, tokenizer, and device setup function
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # Encode questions into embeddings
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    # Saves the list of question embeddings
    torch.save(q_embs, f'{path}/q_embs.pt')

    # Create directory for storing graph data if it doesn't exist
    os.makedirs(path_graphs, exist_ok=True)

    # Process each graph in the dataset
    for index in tqdm(range(len(dataset))):
        # Load nodes and edges data from CSV files created in step_one
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        
        # Fill missing node attributes with empty strings
        # nodes.node_attr.fillna("", inplace=True)
        nodes.fillna({'node_attr': ''}, inplace=True)
        
        # Encode node attributes into embeddings
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # Encode edge attributes into embeddings
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        
        # Create edge index tensor from source and destination nodes
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        # Create PyTorch Geometric (PyG) Data object for the graph
        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        
        # Save the PyG Data object
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')



if __name__ == '__main__':
    step_one()
    step_two()
    generate_split()
