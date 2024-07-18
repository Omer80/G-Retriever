import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'


class WebQSPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def preprocess():
    """
    Preprocess the WebQSP dataset by loading the dataset, nodes, edges, graph files, and question embeddings.
    This function performs the following steps:
    1. Creates directories for cached descriptions and cached graphs if they do not exist.
    2. Loads the WebQSP dataset using the Huggingface datasets package and concatenates train, validation, and test splits.
    3. Loads precomputed question embeddings from the saved file.
    4. Iterates over each graph in the dataset, checks if the graph is already processed, and if not:
       a. Loads the graph, nodes, and edges from their respective files.
       b. Retrieves subgraphs and descriptions via the `retrieval_via_pcst` function.
       c. Saves the subgraph and description to their respective cached directories.

    The function ensures that each graph in the dataset is processed and cached for efficient retrieval and further analysis.
    """

    # Create directories for cached descriptions and graphs if they do not exist
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    # Load the WebQSP dataset and concatenate train, validation, and test splits
    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    # Load precomputed question embeddings
    q_embs = torch.load(f'{path}/q_embs.pt')

    # Iterate over each graph in the dataset
    for index in tqdm(range(len(dataset))):
        # Check if the graph is already processed and cached
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue

        # Load the graph, nodes, and edges from their respective files
        graph = torch.load(f'{path_graphs}/{index}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')

        # Get the question embedding for the current graph
        q_emb = q_embs[index]

        # Retrieve subgraph and description using the retrieval_via_pcst function
        subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)

        # Save the retrieved subgraph and description to their respective cached directories
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    preprocess()

    dataset = WebQSPDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
