import networkx as nx
from keras.models import model_from_json
import pickle
import os
import json
from copy import deepcopy
import pandas
import math
import random
from CGRtools.files import RDFread, RDFwrite, SDFread
from CGRtools.containers import ReactionContainer
from CGRtools.preparer import CGRpreparer
from CGRtools.reactor import CGRreactor
from CIMtools.preprocessing import StandardizeChemAxon
from networkx import DiGraph

path_to_keras_json_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.json"
path_to_keras_h5_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.h5"
path_to_fragmentor = "/home/aigul/Retro/finalize_descriptors.pickle"
path_to_file_with_mol = "/home/aigul/Retro/second_test_for_tree.sdf"
json_file = open(path_to_keras_json_file, 'r')  # load json and create model
loaded_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(loaded_model_json)
keras_model.load_weights(path_to_keras_h5_file)  # load weights into new model
os.environ["PATH"] += ":/opt/fragmentor"
with open(path_to_fragmentor, "rb") as f:
    fr = pickle.load(f)
target = SDFread(path_to_file_with_mol).read()
with SDFread("/home/aigul/Retro/signature/rules.sdf", "r") as f2:
    CGR_env_hyb = f2.read()
with open("/home/aigul/Retro/OldNewNumsOfRules.json") as json_file:
    old_new_nums_of_rules = json.load(json_file)
reverse_dict = {}
for i, j in old_new_nums_of_rules.items():
    reverse_dict[j] = i
with open("/home/aigul/Retro/reagents/reagents_hashes_stand_version.pickle", "rb") as f7:
    reagents_in_store = pickle.load(f7)
with open("/home/aigul/Retro/stand/stand.xml", "r") as stand_file:
    std = stand_file.read()
standardizer = StandardizeChemAxon(std)

reagents_in_store = set(reagents_in_store)
target = standardizer.transform(SDFread(path_to_file_with_mol).read()).as_matrix()[0]


# create array descriptor from molecule
def prep_mol_for_nn(mol_container):
    descriptor = fr.transform([mol_container])
    descriptor_array = descriptor.values
    return descriptor_array


def prediction(descriptor_array):
    prediction = keras_model.predict(descriptor_array)
    return prediction


def return_patterns_rollout(prediction_for_mol):
    u = pandas.DataFrame(prediction_for_mol).iloc[0, :2957].sort_values(ascending=False)[0]
    transform_num = pandas.DataFrame(prediction_for_mol).iloc[0, :2957].sort_values(ascending=False).keys()[0]
    pattern_nums_in_file = []
    pattern_nums_in_file.append(reverse_dict[transform_num])
    return dict(zip(pattern_nums_in_file, [u]))


def return_patterns_expansion(prediction_for_mol):
    number_selected_patterns = 9
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
    for cgr_nums in range(len(cgr_patterns)):
        cgr_pattern = cgr_patterns[cgr_nums]
        cgr_to_reaction = CGRpreparer.decompose(cgr_pattern)
        prod_from_cgr = cgr_to_reaction.products
        reag_from_cgr = cgr_to_reaction.reagents
        if len(reag_from_cgr) < 3:
            react = CGRreactor()
            with RDFwrite("/home/aigul/Retro/templates/test" + str(len(lemon_tree.nodes)) + ".template") as f:
                f.write(ReactionContainer(reagents=[prod_from_cgr[0]], products=reag_from_cgr))
            with RDFread("/home/aigul/Retro/templates/test" + str(len(lemon_tree.nodes)) + ".template", is_template=True) as f:
                template = react.prepare_templates(f.read())
            searcher = react.get_template_searcher(templates=template)
            for i in searcher(molecule):
                destroy = react.patcher(structure=molecule, patch=i.patch)
            try:
                destroy_all.append(standardizer.transform(CGRpreparer.split(destroy)).as_matrix())
                prob_end.append(cgr_prob[cgr_nums])
                for j in range(len(destroy_all)):
                    created_reactions.append(ReactionContainer(reagents=destroy_all[j], products=[molecule]))
                    return destroy_all, created_reactions, prob_end
            except:
                return ([], [], [])


# take ONE molecule, hash
class LemonTree(DiGraph):
    def __init__(self, target):
        super().__init__()
        self.add_node(1, reagents=[target], depth=0, solved=False, reward=0,
                      number_of_visits=0, terminal=False, expanded = False )

    def add_node_(self, parent_node_number, list_of_reagents, probability):
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
        nx.set_node_attributes(self, {node_number: dict_with_attrs})

    def change_atrrs_of_edge(self, parent_node_number, node_number, reaction):
        edge_attrs = {(parent_node_number, node_number): {'reaction': reaction}}
        nx.set_edge_attributes(self, edge_attrs)


    def return_reaction(self, parent_node_number, new_node_number):
        return nx.get_edge_attributes(lemon_tree, "reaction")[parent_node_number, new_node_number]


    def node_depth(self, node_number):
        node_depth_ = self.nodes[node_number]['depth']
        return node_depth_


    @property
    def number_of_visits(self):
        return self.nodes(data="number_of_visits")

    @property
    def node_solved(self):
        return self.nodes(data="solved")


    def go_to_parent(self, node_number):
        try:
            parent = self._pred[node_number]
            return next(iter(parent))
        except:
            return node_number


    def add_edge_(self, parent_node_number, node_number, reaction):
        self.add_edge(parent_node_number, node_number)
        self.change_atrrs_of_edge(parent_node_number=parent_node_number, node_number=node_number, reaction=reaction)


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
        list_of_children = self.go_to_child(node_number)
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

lemon_tree = LemonTree(target)


def return_solutions(new_node_number):
    with RDFwrite("/home/aigul/Retro/templates/first_predictions.rdf") as f:
        global synthetic_path
        parent_node_number = lemon_tree.go_to_parent(new_node_number)
        if new_node_number != parent_node_number:
            synthetic_path.append(lemon_tree.return_reaction(parent_node_number=parent_node_number, new_node_number=new_node_number))
            return_solutions(parent_node_number)
        for solution in synthetic_path:
            f.write(solution)
        return synthetic_path
        synthetic_path = []


def update(new_node_number, reward):
    node_attrs_2 = {"number_of_visits": lemon_tree.nodes[new_node_number]["number_of_visits"] + 1,
                                      "reward": lemon_tree.nodes[new_node_number]["reward"] + (reward)}
    lemon_tree.change_atrrs_of_node(node_number=new_node_number, dict_with_attrs=node_attrs_2)
    Q = (1 / (lemon_tree.nodes[new_node_number]["number_of_visits"])) * (lemon_tree.nodes[new_node_number]["reward"])
    P = lemon_tree.nodes[new_node_number]["probability"]  # prob from NN
    parent_node = lemon_tree.go_to_parent(new_node_number)
    N = lemon_tree.nodes[new_node_number]["number_of_visits"]
    N_pred = lemon_tree.nodes[parent_node]["number_of_visits"]
    a = (Q) + P * (math.sqrt(N_pred) / (1 + N))
    node_attrs_3 = {"Q": Q, "a": a}
    lemon_tree.change_atrrs_of_node(node_number=new_node_number, dict_with_attrs=node_attrs_3)
    if parent_node != 1:
        update(parent_node, reward)


def expansion(node_number, rollout=False):
    global solution_found_counter
    max_depth = 10
    current_node = lemon_tree.go_to_node(node_number)
    list_of_reagents = current_node["reagents"]
    mol_container = list_of_reagents[0]
    if rollout is True:
        new_patterns = return_patterns_rollout(prediction(prep_mol_for_nn(mol_container)))
    else:
        lemon_tree.change_atrrs_of_node(node_number=node_number, dict_with_attrs= {"expanded": True})
        new_patterns = return_patterns_expansion(prediction(prep_mol_for_nn(mol_container)))

    if len(new_patterns) == 0:
        lemon_tree.nodes[node_number]["terminal"] = True
        update(node_number, -1)

    if lemon_tree.nodes[node_number]["terminal"] != True:
        new_mols_from_pred = create_mol_from_pattern(new_patterns, mol_container)
        if len(new_mols_from_pred[0]) == 0:
            lemon_tree.change_atrrs_of_node(node_number=node_number, dict_with_attrs= {"terminal": True} )
            update(node_number, -1)
        else:
            for j2 in range(len(new_mols_from_pred[0])):
                copy_of_list_of_reagents = deepcopy(list_of_reagents[1:])
                # check compounds in DB and if yes exclude it from node molecule list
                for j3 in new_mols_from_pred[0][j2]:
                    if j3.get_signature_hash() not in reagents_in_store:
                        copy_of_list_of_reagents.append(j3)

                # check if node exist if Rollout = False? if not:
                # in rollout == False we add new nodes, 1) if children of nodes absent
                # and 2) if list of hashes != list of new list of hashes; if we have one prediction and this prediction in node we continue
                if rollout is False:
                    if lemon_tree.go_to_child(node_number) == []:
                        new_node_number = lemon_tree.add_node_(list_of_reagents=copy_of_list_of_reagents,
                                                              parent_node_number=node_number,
                                                              probability=new_mols_from_pred[2][j2])

                        lemon_tree.add_edge_(parent_node_number=node_number, node_number=new_node_number,
                                           reaction=new_mols_from_pred[1][j2])

                    elif lemon_tree.nodes[lemon_tree.go_to_child(node_number)[0]]["reagents"] != \
                            copy_of_list_of_reagents:
                        new_node_number = lemon_tree.add_node_(list_of_reagents=copy_of_list_of_reagents,
                                                              parent_node_number=node_number,
                                                              probability=new_mols_from_pred[2][j2])
                        lemon_tree.add_edge_(parent_node_number=node_number, node_number=new_node_number,
                                           reaction=new_mols_from_pred[1][j2])

                    elif lemon_tree.nodes[lemon_tree.go_to_child(node_number)[0]][
                        "reagents"] == copy_of_list_of_reagents:
                        lemon_tree.change_atrrs_of_node(node_number=node_number, dict_with_attrs= {"expanded": True})
                        continue
                    else:
                        continue

                elif rollout is True:
                    if lemon_tree.go_to_child(node_number) == []:
                        new_node_number = lemon_tree.add_node_(list_of_reagents=copy_of_list_of_reagents,
                                                              parent_node_number=node_number,
                                                              probability=new_mols_from_pred[2][j2])

                        lemon_tree.add_edge_(parent_node_number=node_number, node_number=new_node_number,
                                           reaction=new_mols_from_pred[1][j2])
                    else:
                        expansion(node_number)

                if lemon_tree.node_depth(new_node_number) > max_depth and lemon_tree.node_solved(
                        new_node_number) is False:
                    update(new_node_number, -1)
                elif len(lemon_tree.nodes[new_node_number]["reagents"]) == 0:
                    lemon_tree.change_atrrs_of_node(node_number=new_node_number, dict_with_attrs= {"solved": True, "expanded": True, "terminal": True} )
                    solution_found_counter += 1
                    return_solutions(new_node_number)
                    update(new_node_number, 1)
                else:
                    node_nums_rollout.append(new_node_number)
                    expansion(new_node_number, rollout=True)
                    # tut dolzhno bit pusto


def expanded_nodes_in_tree():
    for node in lemon_tree.nodes:
        if not lemon_tree.nodes[node]["expanded"] and lemon_tree.nodes[node]["terminal"]:
            return True


def search(s):
    if len(lemon_tree.nodes) == 1:
        return s
    while lemon_tree.go_to_best_or_random_child(s):
        s = lemon_tree.go_to_best_or_random_child(s)
        if not lemon_tree.nodes[s]["expanded"]:
            return s


def random_search(s):
    if expanded_nodes_in_tree():
        while lemon_tree.go_to_random_child(s):
            s = lemon_tree.go_to_random_child(s)
            if not lemon_tree.nodes[s]["expanded"]:
                return s


def MCTsearch(Max_Iteration=100, Max_Num_Solved=2):
    max_depth = 10

    for i in range(Max_Iteration):
        while solution_found_counter < Max_Num_Solved:
            node_number = 1
            new_node_number = search(node_number) or random_search(node_number)
            if not new_node_number:
                break
            if lemon_tree.node_depth(new_node_number) < max_depth:
                expansion(new_node_number)


synthetic_path = []
solution_found_counter = 0
node_nums_rollout = []

MCTsearch(Max_Iteration=100, Max_Num_Solved=2)

