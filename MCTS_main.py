from copy import deepcopy
import math
from mol_preprocessing_prediction import *

def return_solutions(new_node_number):
    """
    function saves list to rdf file with synthesis ways of target molecule
    """
    with RDFwrite("/home/aigul/Retro/templates/first_predictions.rdf") as f:
        global synthetic_path
        parent_node_number = lemon_tree.go_to_parent(new_node_number)
        if new_node_number != parent_node_number:
            synthetic_path.append(
                lemon_tree.return_reaction(parent_node_number=parent_node_number, new_node_number=new_node_number))
            return_solutions(parent_node_number)
        for solution in synthetic_path:
            f.write(solution)
        return synthetic_path
        synthetic_path = []


def update(new_node_number, reward):
    """
    function updates Q, a, number of visits and reward of node, works recursive until node not  root of tree
    """
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


def expansion(node_number):
    """
    function adds nodes with predicted reactants. if nothing is predicted starts updating node with -1 reward
    :param node_number:
    :return: list of added nodes
    """
    global solution_found_counter
    max_depth = 10
    current_node = lemon_tree.go_to_node(node_number)
    list_of_reagents = current_node["reagents"]
    mol_container = list_of_reagents[0]
    lemon_tree.change_atrrs_of_node(node_number=node_number, dict_with_attrs={"expanded": True})
    new_patterns = return_patterns_expansion(prediction(prep_mol_for_nn(mol_container)))
    if len(new_patterns) == 0:
        lemon_tree.nodes[node_number]["terminal"] = True
        update(node_number, -1)
    if lemon_tree.nodes[node_number]["terminal"] != True:
        new_mols_from_pred = create_mol_from_pattern(new_patterns, mol_container)
        for new_mol in new_mols_from_pred:
            if len(new_mol[0]) == 0:
                lemon_tree.change_atrrs_of_node(node_number=node_number, dict_with_attrs={"terminal": True})
                update(node_number, -1)
            else:
                for j2 in range(len(new_mol[0])):
                    copy_of_list_of_reagents = deepcopy(list_of_reagents[1:])
                    # check compounds in DB and if yes exclude it from node molecule list
                    for j3 in new_mol[0][j2]:
                        if j3.get_signature_hash() not in reagents_in_store:
                            copy_of_list_of_reagents.append(j3)

                    # check if node exist if Rollout = False? if not:
                    # in rollout == False we add new nodes, 1) if children of nodes absent
                    # and 2) if list of hashes != list of new list of hashes; if we have one prediction and this prediction in node we continue
                    if lemon_tree.go_to_child(node_number) == []:
                        new_node_number = lemon_tree.add_node_(list_of_reagents=copy_of_list_of_reagents,
                                                               parent_node_number=node_number,
                                                               probability=new_mol[2][j2])

                        lemon_tree.add_edge_(parent_node_number=node_number, node_number=new_node_number,
                                             reaction=new_mol[1][j2])
                    else:
                        repeated = False
                        for child_num in range(len(lemon_tree.go_to_child(node_number))):
                            if lemon_tree.nodes[lemon_tree.go_to_child(node_number)[child_num]][
                                "reagents"] == copy_of_list_of_reagents:
                                repeated = True
                        if repeated == False:
                            new_node_number = lemon_tree.add_node_(list_of_reagents=copy_of_list_of_reagents,
                                                                   parent_node_number=node_number,
                                                                   probability=new_mol[2][j2])
                            lemon_tree.add_edge_(parent_node_number=node_number, node_number=new_node_number,
                                                 reaction=new_mol[1][j2])
                        else:
                            lemon_tree.change_atrrs_of_node(node_number=node_number, dict_with_attrs={"expanded": True})
                            return lemon_tree.go_to_child(node_number)

                    if lemon_tree.node_depth(new_node_number) > max_depth and lemon_tree.node_solved(
                            new_node_number) is False:
                        update(new_node_number, -1)
                    elif len(lemon_tree.nodes[new_node_number]["reagents"]) == 0:
                        lemon_tree.change_atrrs_of_node(node_number=new_node_number,
                                                        dict_with_attrs={"solved": True, "expanded": True,
                                                                         "terminal": True})
                        solution_found_counter += 1
                        return_solutions(new_node_number)
                        update(new_node_number, 1)
        return lemon_tree.go_to_child(parent_node_number=node_number)


def rollout(node_number):
    """
    function adds nodes with predicted reactants. if nothing is predicted starts updating node with -1 reward, if node
    not terminal recursively starts rollout and ends when model can't predict nothing, or if node has max depth, or
    molecule in list of reagents in store
    :param node_number:
    :return:
    """
    global solution_found_counter
    max_depth = 10
    current_node = lemon_tree.go_to_node(node_number)
    list_of_reagents = current_node["reagents"]
    mol_container = list_of_reagents[0]
    new_patterns = return_patterns_rollout(prediction(prep_mol_for_nn(mol_container)))
    if len(new_patterns) == 0:
        lemon_tree.nodes[node_number]["terminal"] = True
        update(node_number, -1)
    if lemon_tree.nodes[node_number]["terminal"] != True:
        new_mols_from_pred = create_mol_from_pattern(new_patterns, mol_container)
        for new_mol in new_mols_from_pred:
            if len(new_mol[0]) == 0:
                lemon_tree.change_atrrs_of_node(node_number=node_number, dict_with_attrs={"terminal": True})
                update(node_number, -1)
            else:
                for j2 in range(len(new_mol[0])):
                    copy_of_list_of_reagents = deepcopy(list_of_reagents[1:])
                    # check compounds in DB and if yes exclude it from node molecule list
                    for j3 in new_mol[0][j2]:
                        if j3.get_signature_hash() not in reagents_in_store:
                            copy_of_list_of_reagents.append(j3)
                    # check if node exist if Rollout = False? if not:
                    # in rollout == False we add new nodes, 1) if children of nodes absent
                    # and 2) if list of hashes != list of new list of hashes; if we have one prediction and this prediction in node we continue
                    if lemon_tree.go_to_child(node_number) == []:
                        new_node_number = lemon_tree.add_node_(list_of_reagents=copy_of_list_of_reagents,
                                                               parent_node_number=node_number,
                                                               probability=new_mol[2][j2])

                        lemon_tree.add_edge_(parent_node_number=node_number, node_number=new_node_number,
                                             reaction=new_mol[1][j2])
                    else:
                        break
                    if lemon_tree.node_depth(new_node_number) > max_depth and lemon_tree.node_solved(
                            new_node_number) is False:
                        update(new_node_number, -1)
                    elif len(lemon_tree.nodes[new_node_number]["reagents"]) == 0:
                        lemon_tree.change_atrrs_of_node(node_number=new_node_number,
                                                        dict_with_attrs={"solved": True, "expanded": True,
                                                                         "terminal": True})
                        solution_found_counter += 1
                        return_solutions(new_node_number)
                        update(new_node_number, 1)
                    else:
                        node_nums_rollout.append(new_node_number)
                        rollout(new_node_number)




def MCTsearch(Max_Iteration, Max_Num_Solved):
    max_depth = 10
    for i in range(Max_Iteration):
        if solution_found_counter < Max_Num_Solved:
            node_number = 1
            new_node_number = lemon_tree.search(node_number) or lemon_tree.random_search(node_number)
            if not new_node_number:
                continue
            if lemon_tree.node_depth(new_node_number) < max_depth:
                expanded_nodes_nums = expansion(new_node_number)
                if expanded_nodes_nums:
                    for expanded_node in expanded_nodes_nums:
                        rollout(expanded_node)
                else:
                    continue


synthetic_path = []
solution_found_counter = 0
node_nums_rollout = []

MCTsearch(Max_Iteration=100, Max_Num_Solved=2)
