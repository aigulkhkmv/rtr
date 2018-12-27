from networkx import DiGraph
import networkx as nx
from CGRtools.files import SDFread
import random

from networkx import DiGraph
import networkx as nx
from CGRtools.files import SDFread
import random

class LemonTree(DiGraph):
    def __init__(self, target):
        super().__init__()
        self.add_node(1, reagents=[target], depth=0, solved=False, reward=0, 
                      number_of_visits=0, terminal=False, expanded = False )

    def add_node_with_attrs(self, parent_node_number, list_reagents, probability):
        unique_number = len(self.nodes) + 1
        children_depth = int(self.nodes[parent_node_number]['depth']) + 1
        visiting = 0
        solved_ = False
        self.add_node(unique_number)
        nx.set_node_attributes(self, {
            unique_number: {"reagents": list_of_reagents, "depth": children_depth, "solved": solved_,
                            "reward": 0, "number_of_visits": visiting, "terminal": False, "Q": 0,
                            "probability": probability, "a": 0, "expanded": False}})
        return unique_number

    @property
    def node_depth(self):
        return self.nodes(data="depth")

    @property
    def number_of_visits(self):
        return self.nodes(data="number_of_visits")

    @property
    def node_solved(self):
        return self.nodes(data="solved")

    def node_solved(self, node_number):
        return self.nodes[node_number]['solved']

    def go_to_parent(self, node_number):
        try:
            parent = self._pred[node_number]
            return next(iter(parent))
        except:
            return node_number

    def add_edge(self, parent_node_number, node_number, reaction):
        self.add_edge(parent_node_number, node_number)
        edge_attrs = {(parent_node_number, node_number): {'reaction': reaction}}
        nx.set_edge_attributes(self, edge_attrs)

    def go_to_node(self, node_number):
        return self.nodes[node_number]

    def go_to_child(self, parent_node_number):
        y = self.successors(parent_node_number)
        list_of_children = []
        for i in range(len(self.nodes) + 1):
            try:
                list_of_children.append(next(y))
            except:
                break
        return list_of_children

    def go_to_best_or_random_child(self, expanded_node_number):
        E = 0.2
        random_number = random.uniform(0, 1)
        y = self.successors(expanded_node_number)
        list_of_children = []
        for i in range(len(self.nodes) + 1):
            try:
                list_of_children.append(next(y))
            except:
                break
        if random_number < E:
            if len(list_of_children) == 0:
                return None
            else:
                return list_of_children[random.randint(0, len(list_of_children) - 1)]
        if random_number > E:
            best_num = 0
            if len(list_of_children) != 1:
                for num in range(len(list_of_children)):
                    if num + 1 < len(list_of_children):
                        if self.nodes[list_of_children[num]]['a'] < self.nodes[list_of_children[num + 1]]['a']:
                            best_num = num + 1
            else:
                best_num = 0

            if len(list_of_children) == 0:
                return None
            else:
                return list_of_children[best_num]

    def go_to_random_child(self, node_number):
        list_of_children = LemonTree.go_to_child(node_number)
        if len(list_of_children) == 0:
            return None
        else:
            return list_of_children[random.randint(0, len(list_of_children) - 1)]

    def is_terminal(self, parent_node_number):
        y = self.successors(parent_node_number)
        try:
            next(y)
        except:
            return True

path_to_file_with_mol = "/home/aigul/Retro/mol_for_testing_tree.sdf"
with SDFread(path_to_file_with_mol) as f:
    target = next(f)


lt = LemonTree(target)
