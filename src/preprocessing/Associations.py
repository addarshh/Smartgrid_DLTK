# !/usr/bin/env python
# coding: utf-8

import pandas as pd


class Graph:
    """
    : Graph object used to group associated incidents
    :  (network-like, i.e. edges and nodes)
    """

    def __init__(self, v):

        self.V = list(v)
        self.adj = {i: [] for i in self.V}
        return

    def dfs_util(self, temp, v, visited):
        """
        : Helper function to visit nodes in a connected component
        """

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent to this vertex v
        for i in self.adj[v]:
            if not visited[i]:
                temp = self.dfs_util(temp, i, visited)
        return temp

    def add_edge(self, v, w):
        """
        : method to add an undirected edge
        """
        self.adj[v].append(w)
        self.adj[w].append(v)
        return

    def connected_components(self):
        """
        : Method to retrieve connected components in an undirected graph
        """

        visited = {i: False for i in self.V}
        labels = {}
        label = 0
        for v in self.V:
            if not visited[v]:
                temp = []
                inc_group = self.dfs_util(temp, v, visited)
                for inc in inc_group:
                    labels[inc] = label
                label = label + 1
        return labels


def get_associated_incidents(associations_file, event_with_remedy):
    """
    : Pulls association table and filters out the 2 columns that are needed.
    """

    # Load associations table
    association_table = pd.read_csv(associations_file, engine='python')

    # Drop duplicates
    association_table = association_table.drop_duplicates().reset_index(drop=True)
    association_table = association_table[['REQUEST_ID01', 'REQUEST_ID02']]

    # find the incident tickets with association
    association_table['ID01_INC'] = association_table['REQUEST_ID01'].apply(lambda x: 'INC' in x)
    association_table['ID02_INC'] = association_table['REQUEST_ID02'].apply(lambda x: 'INC' in x)

    association_table = association_table[association_table["ID01_INC"] & association_table["ID02_INC"]]

    # remove duplicate entries
    association_table['min_tic'] = association_table[['REQUEST_ID01', 'REQUEST_ID02']].apply(min, axis=1)
    association_table['max_tic'] = association_table[['REQUEST_ID01', 'REQUEST_ID02']].apply(max, axis=1)
    association_table = association_table.drop_duplicates(subset=['min_tic', 'max_tic'])

    placeholder_incidents = event_with_remedy[['INCIDENT_NUMBER']].drop_duplicates()
    placeholder_incidents['min_tic'] = placeholder_incidents['INCIDENT_NUMBER'].copy()
    placeholder_incidents['max_tic'] = placeholder_incidents['INCIDENT_NUMBER'].copy()
    placeholder_incidents['REQUEST_ID01'] = placeholder_incidents['min_tic']
    placeholder_incidents['REQUEST_ID01'] = placeholder_incidents['max_tic']
    placeholder_incidents['ID01_INC'] = True
    placeholder_incidents['ID02_INC'] = True
    placeholder_incidents = placeholder_incidents.drop(columns=["INCIDENT_NUMBER"])

    association_table = association_table.append(placeholder_incidents)

    return association_table


def get_association_labels(associations_file, event_with_remedy):
    """
    : Creates a graph where connected components are associated incidents
    """

    # find set of all INCs
    inc_to_inc_association = get_associated_incidents(associations_file, event_with_remedy)
    inc_set = set(inc_to_inc_association['min_tic'].append(inc_to_inc_association['max_tic']))

    # initialise graph with set of INCs as vertices and each association pair as an edge
    g = Graph(inc_set)
    for idx, row in inc_to_inc_association[['min_tic', 'max_tic']].iterrows():
        g.add_edge(row['min_tic'], row['max_tic'])

    # find association label for each INC using DFS
    labels = g.connected_components()
    assoc_label_df = pd.DataFrame.from_dict(labels, orient='index', columns=['association_label'])

    association_event = event_with_remedy.join(assoc_label_df, on=['INCIDENT_NUMBER'],
                                               how='inner').reset_index(drop=True)

    return association_event
