import networkx as nx
from keras.models import model_from_json
import pickle, os, gc, gzip, json
import sys
import pandas
import shelve
import random
from io import StringIO
from CGRtools.files import RDFread, RDFwrite, SDFread, SDFwrite
from CIMtools.preprocessing import CGR, Fragmentor
from CGRtools.containers import MoleculeContainer, ReactionContainer,QueryContainer
from CGRtools import CGRreactor, CGRpreparer
from CGRtools.preparer import CGRpreparer
from CGRtools.reactor import CGRreactor

# load NN model, fragmentor, target molecule, create networkx graph

path_to_keras_json_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.json"
path_to_keras_h5_file = "/home/aigul/Retro/keras_models/80_200_325_1*2000/model_36_epochs.h5"
path_to_fragmentor = "/home/aigul/Retro/finalize_descriptors.pickle"
path_to_file_with_mol = "/home/aigul/Retro/second_test_for_tree.sdf"

json_file = open(path_to_keras_json_file, 'r')  # load json and create model
loaded_model_json = json_file.read()
json_file.close()
keras_model = model_from_json(loaded_model_json)
keras_model.load_weights(path_to_keras_h5_file)  # load weights into new model

os.environ["PATH"] = "/opt/fragmentor"
with open(path_to_fragmentor, "rb") as f:
    fr = pickle.load(f)

target = SDFread(path_to_file_with_mol).read()

with SDFread("/home/aigul/Retro/signature/SIGNATURE_env_hyb.sdf", "r") as f2:
    CGR_env_hyb = f2.read()

with open("/home/aigul/Retro/OldNewNumsOfRules.json") as json_file:
    old_new_nums_of_rules = json.load(json_file)

reverse_dict = {}
for i, j in old_new_nums_of_rules.items():
    reverse_dict[j] = i

max_depth = 10
G = nx.DiGraph()


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

    print(pattern_nums_in_file, "PATTERN NUMS IN FILE")
    print(pattern_nums_in_file.keys(), "PATTERN NUMS IN FILE .keys()")

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

            # учесть ситуацию, когда нет никаких темплейтов, сейчас он от этого падает!!!

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


def create_hash_from_pred_mols(molecule_from_prediction):
    molecule_hash_list_0 = []
    for i1 in range(len(molecule_from_prediction[0])):
        molecule_hash_list_1 = []
        for j1 in range(len(molecule_from_prediction[0][i1])):
            molecule_hash_list_1.append(str(molecule_from_prediction[0][i1][j1].get_signature_hash()))
        molecule_hash_list_0.append(molecule_hash_list_1)
    reaction_hash_list_0 = []
    for i2 in range(len(molecule_from_prediction[1])):
        reaction_hash_list_0.append(str(molecule_from_prediction[1][i2].get_signature_hash()))

    return molecule_hash_list_0, reaction_hash_list_0, molecule_from_prediction[2]


def save_to_shelve(prediction_hash, prediction):
    db_mols = shelve.open("/home/aigul/Retro/db_mols.txt")
    db_reactions = shelve.open("/home/aigul/Retro/db_reactions.txt")
    for i1 in range(len(prediction_hash[0])):
        for j1 in range(len(prediction_hash[0][i1])):
            try:
                db_mols[str(prediction_hash[0][i1][j1])]
            except KeyError:
                db_mols[str(prediction_hash[0][i1][j1])] = prediction[0][i1][j1]

    for i2 in range(len(prediction_hash[1])):
        print(prediction_hash[1][i2])
        try:
            db_reactions[str(prediction_hash[1][i2])]
        except KeyError:
            db_reactions[str(prediction_hash[1][i2])] = prediction[1][i2]

    db_mols.close()
    db_reactions.close()


# take ONE molecule, hash
class LemonTree():
    def add_node(parent_node_number, molecule_hash, probability):
        unique_number = len(G.nodes) + 1
        children_depth = int(G.nodes[parent_node_number]['depth']) + 1
        visiting = 0
        solved_ = False
        #        Check molecule in database. Now I have't database => solved:"False"
#                 db_mols = shelve.open("/home/aigul/Retro/db_mols.txt")
#                 mol = db_mols[self.molecule_hash]
        #         if mol in db:
        #             solved_ = True
        #             else:
        #             solved_ = False
        #         db_mols.close()
        G.add_node(unique_number)  # add some unique number as node name   WRONG
        node_attrs = {unique_number: {"list_of_molecule_hash":molecule_hash,"depth":children_depth,"solved":solved_,
                                      "reward":0,"number_of_visits":visiting,"terminal":False,"Q":0,
                                      "probability": probability, "a":0}}
        nx.set_node_attributes(G, node_attrs)
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
            print("This is root")
            return node_number

    def add_edge(parent_node_number, node_number, reaction_hash):
        G.add_edge(parent_node_number, node_number)
        edge_attrs = {(parent_node_number, node_number): {'reaction_hash': reaction_hash}}
        nx.set_edge_attributes(G, edge_attrs)

    def go_to_node(node_number):
        return G.nodes[node_number]

    def go_to_child(parent_node_number):
        y = G.successors(parent_node_number)
        list_of_childs = []
        for i in range(len(G.nodes) + 1):
            try:
                list_of_childs.append(next(y))
            except:
                break

        return list_of_childs

    def go_to_best_or_random_child(parent_node_number):
        E = 0.2
        random_number = random.uniform(0, 1)
        y = G.successors(parent_node_number)
        list_of_childs = []
        for i in range(len(G.nodes) + 1):
            try:
                list_of_childs.append(next(y))
            except:
                break

        if random_number < E:
            return list_of_childs[random.randint(0, len(list_of_childs) - 1)]

        if random_number > E:
            best_num = 0
            for num in range(len(list_of_childs)):
                if num + 1 < len(list_of_childs):
                    if G.nodes[num]['a'] > G.nodes[num + 1]['a']:    #was reward, but it was wrong
                        best_num = num
                    else:
                        best_num = num+1
            return list_of_childs[best_num]

    def is_terminal(parent_node_number):
        y = G.successors(parent_node_number)
        try:
            next(y)
        except:
            return True
            print("Terminal node")


def search(node_number):  # traverse the tree
    max_depth = 10

    if len(G.nodes) == 1:  # or we have IndexError for root
        print("I was here")
        new_node_number = expansion(node_number)
        print("I return", new_node_number)

    elif len(G.nodes) != 1:
        print("In search i go elif")
        LemonTree.go_to_best_or_random_child(1)
    #         return new_node_num

    if LemonTree.number_of_visits(new_node_num) > 1 and G.nodes[new_node_num]["terminal"] == False:
        search(new_node_num)
    elif G.nodes[new_node_num]["terminal"] == True:
        node_attrs = {node_number: {"reward": G.nodes[node_number]["reward"] + (-1)}}
        nx.set_node_attributes(G, node_attrs)
        update(new_node_num)
    else:
        expansion(new_node_num)

    print("SEARCH", new_node_number)

    if solution_found_counter > 0:
        return_solutions(new_node_number)  # WRITE THIS FUNCTION

    elif LemonTree.node_depth(new_node_num) > max_depth:
        print("STOP!")
        sys.exit(0)

    elif solution_found_counter == 1:
        search(1)


synthetic_path = []
def return_solutions(new_node_number):
    with RDFwrite("/home/aigul/Retro/templates/first_predictions.rdf") as f:
        print("RETURN SOLUTIONS:", new_node_number)
        global synthetic_path
        print(synthetic_path)
        parent_node_number = LemonTree.go_to_parent(new_node_number)
        if new_node_number != parent_node_number:
            synthetic_path.append(nx.get_edge_attributes(G,'reaction_hash')[parent_node_number, new_node_number])
            print("RETURN SOLUTIONS")
            print(parent_node_number, new_node_number)
            print("RETURN SOLUTIONS", nx.get_edge_attributes(G,'reaction_hash')[parent_node_number, new_node_number])
            return_solutions(parent_node_number)

        for solution in synthetic_path:

            db_reactions = shelve.open("/home/aigul/Retro/db_reactions.txt")
            rea_container = db_reactions[str(solution)]
            db_mols.close()
            f.write(rea_container)

        return synthetic_path
        synthetic_path = []


def update(new_node_number, reward):
    node_attrs_2 = {new_node_number: {"number_of_visits": G.nodes[new_node_number]["number_of_visits"] + 1,
                                  "reward": G.nodes[new_node_number]["reward"]+(reward)}}
    nx.set_node_attributes(G, node_attrs_2)
    Q = (1/(G.nodes[new_node_number]["number_of_visits"]))*(G.nodes[new_node_number]["reward"])
    P = G.nodes[new_node_number]["probability"] #prob from NN
    parent_node = LemonTree.go_to_parent(new_node_number)
    N = G.nodes[new_node_number]["number_of_visits"]
    N_pred = G.nodes[parent_node]["number_of_visits"]
    a = (Q/N) + P*(math.sqrt(N_pred)/(1+N))
    node_attrs_3 = {new_node_number: {"Q": Q, "a": a}}
    nx.set_node_attributes(G, node_attrs_3)
    if parent_node != 1:
        update(parent_node, reward)


def expansion(node_number, rollout=False):
    global solution_found_counter
    max_depth = 10
    db_mol = shelve.open("/home/aigul/Retro/db_mols.txt")
    current_node = LemonTree.go_to_node(node_number)
    list_of_molecular_hash = current_node["list_of_molecule_hash"]

    print("EXPANSION", list_of_molecular_hash)

    mol_container = db_mol[str(list_of_molecular_hash.pop(0))]
    if rollout == True:
        new_patterns = return_patterns_rollout(prediction(prep_mol_for_nn(mol_container)))
        print("I go to roll-out")
        print(new_patterns)
    else:
        new_patterns = return_patterns_expansion(prediction(prep_mol_for_nn(mol_container)))
        print("I go to expansion")
        print(new_patterns)
    if len(new_patterns) == 0:
        G.nodes[node_number]["terminal"] = True
    print("I go to create mols")
    if G.nodes[node_number]["terminal"] != True:
        new_mols_from_pred = create_mol_from_pattern(new_patterns, mol_container)
        print("new_mols_from_pred")
        new_hash_from_pred = create_hash_from_pred_mols(new_mols_from_pred)
        save_to_shelve(prediction=new_mols_from_pred, prediction_hash=new_hash_from_pred)

        for j2 in range(len(new_hash_from_pred[0])):
            copy_of_list_of_molecular_hash = list_of_molecular_hash
            for j3 in new_hash_from_pred[0][j2]:
                print(j3, type(j3), "j3 from EXPANSION")

                # check in data base, THIS BASE ONLY FOR ONE MOLECULE
                sigma_aldrich_db = shelve.open("/home/aigul/Retro/test_db/sigma_aldrich_db.txt")
                try:
                    sigma_aldrich_db[j3]
                except:
                    copy_of_list_of_molecular_hash.append(j3)
                sigma_aldrich_db.close()
            new_node_number = LemonTree.add_node(molecule_hash=copy_of_list_of_molecular_hash,
                                                 parent_node_number=node_number, probability=new_hash_from_pred[2][j2])
            LemonTree.add_edge(parent_node_number=node_number, node_number=new_node_number,
                               reaction_hash=new_hash_from_pred[1][j2])
            print("I add node and edge!")

            if LemonTree.node_depth(new_node_number) > max_depth and LemonTree.node_solved(new_node_number) == False:
                #                 node_attrs_0 = {new_node_number: {"reward": G.nodes[new_node_number]["reward"] + (-1)}}
                #                 nx.set_node_attributes(G, node_attrs_0)
                print("1")
                update(new_node_number, -1)
                # go to new pattern
            elif len(G.nodes[new_node_number]["list_of_molecule_hash"]) == 0:
                #                 node_attrs_1 = {new_node_number: {"reward": G.nodes[new_node_number]["reward"] + 1, "solved": True}}
                #                 nx.set_node_attributes(G, node_attrs_1)
                node_attrs_1 = {new_node_number: {"solved": True}}
                nx.set_node_attributes(G, node_attrs_1)
                solution_found_counter += 1
                print("2")
                update(new_node_number, 1)
                return new_node_number
            else:
                expansion(new_node_number, rollout=True)
                print("3")
        # tut dolzhno bit pusto


# for target molecule
db_mols = shelve.open("/home/aigul/Retro/db_mols.txt")
db_reactions = shelve.open("/home/aigul/Retro/db_reactions.txt")
target_mol_hash=target[0].get_signature_hash()
db_mols[str(target_mol_hash)] = target[0]
db_mols.close()

# add target molecule at graph
tar = target[0]
molecule_hash = str(tar.get_signature_hash())
G.add_node(1)
attrs_1 = {1: {"list_of_molecule_hash": [molecule_hash], "depth": 0, "solved": False, "reward": 0, "number_of_visits": 0,
             "terminal": False}}
nx.set_node_attributes(G, attrs_1)

search(1)
