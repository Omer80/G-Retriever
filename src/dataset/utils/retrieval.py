import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data


def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    """
    Extracts a subgraph from the given graph based on the relevance of nodes and edges to a query embedding using the Prize-Collecting Steiner Tree (PCST) algorithm.

    This function assigns higher "prizes" to nodes and edges that exhibit a stronger relevance to the query, as determined by their cosine similarity to the query embedding. The objective is to identify a subgraph that optimizes the total prize of nodes and edges, minus the costs associated with the size of the subgraph.

    Args:
        graph (Data): The input graph, a `torch_geometric.data.Data` object, containing node features (`x`), edge indices (`edge_index`), and edge features (`edge_attr`).
        q_emb (torch.Tensor): The embedding of the query, used to compute similarity with nodes and edges.
        textual_nodes (pd.DataFrame): DataFrame containing textual information about the nodes.
        textual_edges (pd.DataFrame): DataFrame containing textual information about the edges.
        topk (int, optional): The number of top nodes to consider based on their similarity to the query. Defaults to 3.
        topk_e (int, optional): The number of top edges to consider based on their similarity to the query. Defaults to 3.
        cost_e (float, optional): The base cost assigned to edges for the PCST problem. Defaults to 0.5.

    Returns:
        Data: A `torch_geometric.data.Data` object containing the extracted subgraph.
        str: A description of the extracted subgraph including selected nodes and edges in CSV format.
    """
    import torch
    import numpy as np
    from torch_geometric.data import Data
    from pcst_fast import pcst_fast

    # Constant for edge prize adjustment
    c = 0.01

    # If there are no textual nodes or edges, return the original graph and a description
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Define PCST parameters
    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    # Calculate node prizes based on cosine similarity to query embedding
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    # Calculate edge prizes based on cosine similarity to query embedding
    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e - k) / sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - c)
        # Adjust edge cost to ensure at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    # Prepare data for PCST algorithm
    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    # Combine real and virtual prizes and edges
    prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)

    # Solve PCST problem
    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    # Extract selected nodes and edges
    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    # Create subgraph description
    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    # Map nodes to new indices
    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    # Create new graph data
    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc