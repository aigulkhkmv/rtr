import networkx as nx
from keras.models import model_from_json
import pickle
import os
import json
from copy import deepcopy
import pandas
import shelve
import math
import random
from CGRtools.files import RDFread, RDFwrite, SDFread
from CGRtools.containers import ReactionContainer
from CGRtools.preparer import CGRpreparer
from CGRtools.reactor import CGRreactor

path_to_keras_json_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.json"
path_to_keras_h5_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.h5"
path_to_fragmentor = "/home/aigul/Retro/finalize_descriptors.pickle"
path_to_file_with_mol = "/home/aigul/Retro/mol_for_testing_tree.sdf"
json_file = open(path_to_keras_json_file, 'r')  # load json and create model
loaded_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(loaded_model_json)
keras_model.load_weights(path_to_keras_h5_file)  # load weights into new model
os.environ["PATH"] = "/opt/fragmentor"
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
path_to_file_with_mol = "/home/aigul/Retro/second_test_for_tree.sdf"
target=SDFread(path_to_file_with_mol).read()
solution_found_counter = 0


# create array descriptor from molecule
def prep_mol_for_nn(mol_container):
    descriptor =fr.transform([mol_container])
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
            with RDFwrite("/home/aigul/Retro/templates/test" + str(len(G.nodes)) + ".template") as f:
                f.write(ReactionContainer(reagents=[prod_from_cgr[0]], products=reag_from_cgr))
            with RDFread("/home/aigul/Retro/templates/test" + str(len(G.nodes)) + ".template", is_template=True) as f:
                template = react.prepare_templates(f.read())
            searcher = react.get_template_searcher(templates=template)
            for i in searcher(molecule):
                destroy = react.patcher(structure=molecule, patch=i.patch)
            try:
                destroy_all.append(CGRpreparer.split(destroy))
                prob_end.append(cgr_prob[cgr_nums])

                for j in range(len(destroy_all)):
                    created_reactions.append(ReactionContainer(reagents=destroy_all[j], products=[molecule]))
                    return destroy_all, created_reactions, prob_end
            except:
                return ([], [], [])


# take ONE molecule, hash
class LemonTree():
    def add_node_(parent_node_number, list_of_molecules, probability):
        unique_number = len(G.nodes) + 1
        children_depth = int(G.nodes[parent_node_number]['depth']) + 1
        visiting = 0
        solved_ = False
        G.add_node(unique_number)
        nx.set_node_attributes(G, {
            unique_number: {"list_of_molecules": list_of_molecules, "depth": children_depth, "solved": solved_,
                            "reward": 0, "number_of_visits": visiting, "terminal": False, "Q": 0,
                            "probability": probability, "a": 0, "expanded": False}})
        return unique_number

    def node_depth(node_number):
        node_depth_ = G.nodes[node_number]['depth']
        return node_depth_

    def number_of_visits(node_number):
        number_of_visits_ = G.nodes[node_number]['number_of_visits']
        return number_of_visits_

    def node_solved(node_number):
        node_solved_ = G.nodes[node_number]['solved']
        return node_solved_

    def go_to_parent(node_number):
        try:
            parent = G._pred[node_number]
            return next(iter(parent))
        except:
            return node_number

    def add_edge(parent_node_number, node_number, reaction):
        G.add_edge(parent_node_number, node_number)
        edge_attrs = {(parent_node_number, node_number): {'reaction': reaction}}
        nx.set_edge_attributes(G, edge_attrs)

    def go_to_node(node_number):
        return G.nodes[node_number]

    def go_to_child(parent_node_number):
        y = G.successors(parent_node_number)
        list_of_children = []
        for i in range(len(G.nodes) + 1):
            try:
                list_of_children.append(next(y))
            except:
                break
        return list_of_children

    def go_to_best_or_random_child(expanded_node_number):
        E = 0.2
        random_number = random.uniform(0, 1)
        y = G.successors(expanded_node_number)
        list_of_children = []
        for i in range(len(G.nodes) + 1):
            try:
                list_of_children.append(next(y))
            except:
                break
        if random_number < E:
            return list_of_children[random.randint(0, len(list_of_children) - 1)]
        if random_number > E:
            best_num = 0
            for num in range(len(list_of_children)):
                if num + 1 < len(list_of_children):
                    if G.nodes[list_of_children[num]]['a'] < G.nodes[list_of_children[num + 1]]['a']:
                        best_num = num
                else:
                    best_num = 0
            return list_of_children[best_num]

    def go_to_random_child(root_number):
        print("I here")
        list_of_children = LemonTree.go_to_child(root_number)
        print(list_of_children)
        return list_of_children[random.randint(0, len(list_of_children) - 1)]

    def is_terminal(parent_node_number):
        y = G.successors(parent_node_number)
        try:
            next(y)
        except:
            return True

synthetic_path = []

def return_solutions(new_node_number):
    with RDFwrite("/home/aigul/Retro/templates/first_predictions.rdf") as f:
        global synthetic_path
        parent_node_number = LemonTree.go_to_parent(new_node_number)
        if new_node_number != parent_node_number:
            synthetic_path.append(nx.get_edge_attributes(G, 'reaction')[parent_node_number, new_node_number])
            return_solutions(parent_node_number)
        for solution in synthetic_path:
            f.write(rea_container)
        return synthetic_path
        synthetic_path = []


def update(new_node_number, reward):
    node_attrs_2 = {new_node_number: {"number_of_visits": G.nodes[new_node_number]["number_of_visits"] + 1,
                                  "reward": G.nodes[new_node_number]["reward"]+(reward)}}
    nx.set_node_attributes(G, node_attrs_2)
    Q = (1/(G.nodes[new_node_number]["number_of_visits"]))*(G.nodes[new_node_number]["reward"])
    P = G.nodes[new_node_number]["probability"]               #prob from NN
    parent_node = LemonTree.go_to_parent(new_node_number)
    N = G.nodes[new_node_number]["number_of_visits"]
    N_pred = G.nodes[parent_node]["number_of_visits"]
    a = (Q) + P*(math.sqrt(N_pred)/(1+N))
    node_attrs_3 = {new_node_number: {"Q": Q, "a": a}}
    nx.set_node_attributes(G, node_attrs_3)
    if parent_node != 1:
        update(parent_node, reward)


def expansion(node_number, rollout=False):
    global solution_found_counter
    max_depth = 10
    current_node = LemonTree.go_to_node(node_number)
    list_of_molecules = current_node["list_of_molecules"]
    mol_container = list_of_molecules[0]
    if rollout is True:
        new_patterns = return_patterns_rollout(prediction(prep_mol_for_nn(mol_container)))
    else:
        nx.set_node_attributes(G, {node_number: {"expanded": True}})
        new_patterns = return_patterns_expansion(prediction(prep_mol_for_nn(mol_container)))

    if len(new_patterns) == 0:
        G.nodes[node_number]["terminal"] = True
        update(node_number, -1)

    if G.nodes[node_number]["terminal"] != True:
        new_mols_from_pred = create_mol_from_pattern(new_patterns, mol_container)
        if len(new_mols_from_pred[0]) == 0:
            nx.set_node_attributes(G, {node_number: {"terminal": True}})
            update(node_number, -1)
        else:
            for j2 in range(len(new_mols_from_pred[0])):
                copy_of_list_of_molecules = deepcopy(list_of_molecules[1:])
                # check compounds in DB and if yes exclude it from node molecule list
                for j3 in new_mols_from_pred[0][j2]:
                    sigma_aldrich_db = shelve.open("/home/aigul/Retro/test_db/sigma_aldrich_db.txt")
                    try:
                        sigma_aldrich_db[j3]
                    except:
                        copy_of_list_of_molecules.append(j3)
                    sigma_aldrich_db.close()

                # check if node exist if Rollout = False? if not:
                # in rollout == False we add new nodes, 1) if children of nodes absent
                # and 2) if list of hashes != list of new list of hashes; if we have one prediction and this prediction in node we continue
                if rollout is False:
                    if LemonTree.go_to_child(node_number) == []:
                        new_node_number = LemonTree.add_node_(list_of_molecules=copy_of_list_of_molecules,
                                                              parent_node_number=node_number,
                                                              probability=new_mols_from_pred[2][j2])

                        LemonTree.add_edge(parent_node_number=node_number, node_number=new_node_number,
                                           reaction=new_mols_from_pred[1][j2])

                    elif G.nodes[LemonTree.go_to_child(node_number)[0]]["list_of_molecules"] != \
                            copy_of_list_of_molecules:
                        new_node_number = LemonTree.add_node_(list_of_molecules=copy_of_list_of_molecules,
                                                              parent_node_number=node_number,
                                                              probability=new_mols_from_pred[2][j2])
                        LemonTree.add_edge(parent_node_number=node_number, node_number=new_node_number,
                                           reaction=new_mols_from_pred[1][j2])

                    elif G.nodes[LemonTree.go_to_child(node_number)[0]][
                        "list_of_molecules"] == copy_of_list_of_molecules:
                        node_attrs_7 = {node_number: {"expanded": True}}
                        nx.set_node_attributes(G, node_attrs_7)
                        continue
                    else:
                        continue

                elif rollout is True:
                    if LemonTree.go_to_child(node_number) == []:
                        new_node_number = LemonTree.add_node_(list_of_molecules=copy_of_list_of_molecules,
                                                              parent_node_number=node_number,
                                                              probability=new_mols_from_pred[2][j2])

                        LemonTree.add_edge(parent_node_number=node_number, node_number=new_node_number,
                                           reaction=new_mols_from_pred[1][j2])
                    else:
                        expansion(node_number)

                if LemonTree.node_depth(new_node_number) > max_depth and LemonTree.node_solved(new_node_number) is False:
                    update(new_node_number, -1)
                elif len(G.nodes[new_node_number]["list_of_molecules"]) == 0:
                    node_attrs_1 = {new_node_number: {"solved": True, "expanded": True, "terminal": True}}
                    nx.set_node_attributes(G, node_attrs_1)
                    solution_found_counter += 1
                    return_solutions(new_node_number)
                    update(new_node_number, 1)
                else:
                    node_nums_rollout.append(new_node_number)
                    expansion(new_node_number, rollout=True)


def search(node_number, random=False):  # traverse the tree
    if len(G.nodes) == 1:  # or we have IndexError for root
        return node_number
    if random is False:
        new_node_number_ = LemonTree.go_to_best_or_random_child(node_number)
        if G.nodes[node_number]["expanded"] is True and G.nodes[new_node_number_]["expanded"] is False:
            new_node_number = new_node_number_
        else:
            return search(new_node_number_, False)
    else:
        new_node_number = LemonTree.go_to_random_child(1)
    if G.nodes[new_node_number]["expanded"] and not G.nodes[new_node_number]["terminal"]:
        print("search")
        search(new_node_number, random)
    elif not G.nodes[new_node_number]["expanded"] and not G.nodes[new_node_number]["terminal"]:
        return new_node_number
    elif G.nodes[new_node_number]["terminal"]:
        print("Let's start random")
        search(1, random=True)


def MCTsearch(Max_Iteration=100, Max_Num_Solved=2):
    max_depth = 10
    for i in range(Max_Iteration):
        while solution_found_counter < Max_Num_Solved:
            node_number = 1
            new_node_number = search(node_number)
            if LemonTree.node_depth(new_node_number) < max_depth:
                expansion(new_node_number)


node_nums_rollout = []
G = nx.DiGraph()
# add target molecule at graph
tar = target[0]
target_molecule = tar
G.add_node(1)
attrs_1 = {1: {"list_of_molecules": [target_molecule], "depth": 0, "solved": False, "reward": 0, "number_of_visits": 0,
             "terminal": False}}
nx.set_node_attributes(G, attrs_1)

MCTsearch(Max_Iteration=100, Max_Num_Solved=2)
