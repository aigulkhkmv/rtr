import networkx as nx
from keras.models import model_from_json
import pickle
import os
import json
import pandas
import random
from CGRtools.files import RDFread, RDFwrite, SDFread
from CGRtools.containers import ReactionContainer
from CGRtools.preparer import CGRpreparer
from CGRtools.reactor import CGRreactor
from CIMtools.preprocessing import StandardizeChemAxon
from networkx import DiGraph

# load target molecule
path_to_file_with_mol = "/home/aigul/Retro/reagents/target0.sdf"

#load fragmentor
path_to_fragmentor = "/home/aigul/Retro/finalize_descriptors.pickle"
os.environ["PATH"] += ":/opt/fragmentor"
with open(path_to_fragmentor, "rb") as f:
    fr = pickle.load(f)

# load reagents in store
with open("/home/aigul/Retro/reagents/special_base.pickle", "rb") as f7:
    reagents_in_store = pickle.load(f7)
reagents_in_store = set(reagents_in_store)


# create reverse dict with rules
with SDFread("/home/aigul/Retro/signature/rules.sdf", "r") as f2:
    CGR_env_hyb = f2.read()
with open("/home/aigul/Retro/OldNewNumsOfRules.json") as json_file:
    old_new_nums_of_rules = json.load(json_file)
reverse_dict = {}
for i, j in old_new_nums_of_rules.items():
    reverse_dict[j] = i

# load Keras models for predictions
path_to_keras_json_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.json"
path_to_keras_h5_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.h5"
json_file = open(path_to_keras_json_file, 'r')  # load json and create model
loaded_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(loaded_model_json)
keras_model.load_weights(path_to_keras_h5_file)  # load weights into new model

# load ChemAxon standardizer
with open("/home/aigul/Retro/stand/stand.xml", "r") as stand_file:
    std = stand_file.read()
standardizer = StandardizeChemAxon(std)


def prep_mol_for_nn(mol_container):
    """
    function creates array descriptor from molecule
    :param mol_container:
    :return: array descriptor
    """
    descriptor = fr.transform([mol_container])
    descriptor_array = descriptor.values
    return descriptor_array


def prediction(descriptor_array):
    """
    function makes predictions with Keras neural network model
    :param descriptor_array:
    :return: vector with probabilities for each rule
    """
    prediction = keras_model.predict(descriptor_array)
    return prediction


def return_patterns_rollout(prediction_for_mol):
    """
    function finds pattern number in file with patterns for rollout procedure
    :param prediction_for_mol: vector with probabilities for each rule
    :return: dict with one number of pattern with best probability as key and probability
    """
    u = pandas.DataFrame(prediction_for_mol).iloc[0, :2957].sort_values(ascending=False)[0]
    transform_num = pandas.DataFrame(prediction_for_mol).iloc[0, :2957].sort_values(ascending=False).keys()[0]
    pattern_nums_in_file = []
    pattern_nums_in_file.append(reverse_dict[transform_num])
    return dict(zip(pattern_nums_in_file, [u]))


def return_patterns_expansion(prediction_for_mol):
    """
    function finds patterns number in file with patterns expansion procedure
    :param prediction_for_mol: vector with probabilities for each rule
    :return: dict with numbers of pattern and probability. take patterns that give probability = 0.9, max 10 patterns (you can change it))
    """
    number_selected_patterns = 10
    u = pandas.DataFrame(prediction_for_mol).iloc[0, :2957].sort_values(ascending=False)
    count = 0
    sum_u = 0
    transform_num = []
    transform_prob = []
    for j, i in enumerate(list(u.values)):
        count += 1
        sum_u += i
        transform_prob.append(i)
        transform_num.append(list(u.keys())[j])
        if len(transform_num) > number_selected_patterns:
            break
        elif sum_u >= 0.9:
            break
    pattern_nums_in_file = []
    for new_pattern_num in transform_num:
        pattern_nums_in_file.append(reverse_dict[new_pattern_num])
    return dict(zip(pattern_nums_in_file, transform_prob))


def create_mol_from_pattern(pattern_nums_in_file, molecule):
    cgr_patterns = []
    cgr_prob = []
    prob_end = []
    destroy_all = []
    created_reactions = []
    for cgr in CGR_env_hyb:
        if cgr.meta["id"] in pattern_nums_in_file.keys():
            cgr_patterns.append(cgr)
            cgr_prob.append(pattern_nums_in_file[cgr.meta["id"]])

    reagents_reactions_probs = []
    for cgr_nums in range(len(cgr_patterns)):
        cgr_pattern = cgr_patterns[cgr_nums]
        cgr_to_reaction = CGRpreparer.decompose(cgr_pattern)
        prod_from_cgr = cgr_to_reaction.products
        reag_from_cgr = cgr_to_reaction.reagents
        if len(reag_from_cgr) < 3:
            react = CGRreactor()
            with RDFwrite("/home/aigul/Retro/templates/test" + str(len(lemon_tree.nodes)) + ".template") as f:
                f.write(ReactionContainer(reagents=[prod_from_cgr[0]], products=reag_from_cgr))
            with RDFread("/home/aigul/Retro/templates/test" + str(len(lemon_tree.nodes)) + ".template",
                         is_template=True) as f:
                template = react.prepare_templates(f.read())
            searcher = react.get_template_searcher(templates=template)
            for i in searcher(molecule):
                destroy = react.patcher(structure=molecule, patch=i.patch)
            try:
                destroy_all.append(CGRpreparer.split(destroy))
                prob_end.append(cgr_prob[cgr_nums])
                for j in range(len(destroy_all)):
                    created_reactions.append(ReactionContainer(reagents=destroy_all[j], products=[molecule]))
                    created_reactions = standardizer.transform(created_reactions).as_matrix()
                    reagents_reactions_probs.append([[created_reactions[0].reagents], created_reactions, prob_end])
            except:
                reagents_reactions_probs.append([[], [], []])
    return reagents_reactions_probs


class LemonTree(DiGraph):
    """
    class LemonTree DiGraph from networkx
    """
    def __init__(self, target):
        super().__init__()
        self.add_node(1, reagents=[target], depth=0, solved=False, reward=0,
                      number_of_visits=0, terminal=False, expanded=False)

    def add_node_(self, parent_node_number, list_of_reagents, probability):
        """
        function adds node with attributes in LemonTree
        :return: new node number
        """
        unique_number = len(self.nodes) + 1
        children_depth = int(self.nodes[parent_node_number]['depth']) + 1
        visiting = 0
        solved_ = False
        self.add_node(unique_number)
        dict_with_atrrs = {"reagents": list_of_reagents, "depth": children_depth,
                           "solved": solved_, "reward": 0, "number_of_visits": visiting,
                           "terminal": False, "Q": 0, "probability": probability, "a": 0,
                           "expanded": False}
        self.change_atrrs_of_node(unique_number, dict_with_atrrs)
        return unique_number

    def change_atrrs_of_node(self, node_number, dict_with_attrs):
        """
        function changes node attributes
        """
        nx.set_node_attributes(self, {node_number: dict_with_attrs})

    def change_atrrs_of_edge(self, parent_node_number, node_number, reaction):
        """
        function changes edge attributes
        """
        edge_attrs = {(parent_node_number, node_number): {'reaction': reaction}}
        nx.set_edge_attributes(self, edge_attrs)

    def return_reaction(self, parent_node_number, new_node_number):
        """
        function returns reaction in edge between 2 nodes
        """
        return nx.get_edge_attributes(lemon_tree, "reaction")[parent_node_number, new_node_number]

    def node_depth(self, node_number):
        """
        function returns node depth
        """
        node_depth_ = self.nodes[node_number]['depth']
        return node_depth_

    @property
    def number_of_visits(self):
        """
        function returns number of visits of node
        """
        return self.nodes(data="number_of_visits")

    @property
    def node_solved(self):
        """
        function returns True if node solved and False if not solved
        """
        return self.nodes(data="solved")

    def go_to_parent(self, node_number):
        """
        function returns parent node number, if parent node absent, return original node number
        """
        try:
            parent = self._pred[node_number]
            return next(iter(parent))
        except:
            return node_number

    def add_edge_(self, parent_node_number, node_number, reaction):
        """
        function adds edge between 2 nodes and in edge adds reaction
        """
        self.add_edge(parent_node_number, node_number)
        self.change_atrrs_of_edge(parent_node_number=parent_node_number, node_number=node_number, reaction=reaction)

    def go_to_node(self, node_number):
        return self.nodes[node_number]

    def go_to_child(self, parent_node_number):
        """
        function returns list of children of parent node, if children are absent function is breaking
        """
        y = self.successors(parent_node_number)
        list_of_children = []
        for i in range(len(self.nodes) + 1):
            try:
                list_of_children.append(next(y))
            except:
                break
        return list_of_children

    def go_to_best_or_random_child(self, expanded_node_number):
        """
        if E < some number function returns random child;
        else function returns child with best "a"
        :return: list with best or random child
        """
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
        """
        function returns random child
        """
        list_of_children = self.go_to_child(node_number)
        if len(list_of_children) == 0:
            return None
        else:
            return list_of_children[random.randint(0, len(list_of_children) - 1)]

    def is_terminal(self, parent_node_number):
        """
        function returns True if node is terminal
        """
        y = self.successors(parent_node_number)
        try:
            next(y)
        except:
            return True

    def expanded_nodes_in_tree(self):
        """
        return True if
        :return:
        """
        for node in self.nodes:
            if not self.nodes[node]["expanded"] and self.nodes[node]["terminal"]:
                return True

    def search(self, node):
        """
        function returns original node if it is a root else returns best or random child
        """
        if len(self.nodes) == 1:
            return node
        while self.go_to_best_or_random_child(node):
            node = self.go_to_best_or_random_child(node)
            if not self.nodes[node]["expanded"] and not self.nodes[node]["terminal"]:
                return node

    def random_search(self, node):
        """
        if algorithm go to terminal node, must begining random search
        """
        if self.expanded_nodes_in_tree():
            while self.go_to_random_child(node):
                node = self.go_to_random_child(node)
                if not self.nodes[node]["expanded"]:
                    return node


target = standardizer.transform(SDFread(path_to_file_with_mol).read()).as_matrix()[0]
lemon_tree = LemonTree(target)

