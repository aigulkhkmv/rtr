import networkx as nx
from keras.models import model_from_json
import pickle
import os
import gzip
import json
import pandas
import shelve
from CGRtools.files import RDFread, RDFwrite, SDFread, SDFwrite
from CGRtools.containers import MoleculeContainer, ReactionContainer, QueryContainer
from CGRtools.preparer import CGRpreparer
from CGRtools.reactor import CGRreactor

# load NN model, fragmentor, target molecule, create networkx graph
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

with SDFread("/home/aigul/Retro/signature/SIGNATURE_env_hyb.sdf", "r") as f2:
    CGR_env_hyb = f2.read()

with open("/home/aigul/Retro/OldNewNumsOfRules.json") as json_file:
    old_new_nums_of_rules = json.load(json_file)

reverse_dict = {}
for i, j in old_new_nums_of_rules.items():
    reverse_dict[j] = i

G = nx.DiGraph()


# create array descriptor from molecule
def prep_mol_for_nn(mol_container):
    descriptor = fr.transform(mol_container)
    descriptor_array = descriptor.values
    return descriptor_array


def prediction(descriptor_array):
    prediction = keras_model.predict(descriptor_array)
    return prediction


def returns_patterns(prediction_for_mol):
    number_selected_patterns = 9
    u = pandas.DataFrame(prediction_for_mol).iloc[0, :2957].sort_values(ascending=False)

    count = 0
    sum_u = 0

    transform_num = []

    for j, i in enumerate(list(u.values)):
        count += 1
        sum_u += i
        transform_num.append(list(u.keys())[j])

        if len(transform_num) > number_selected_patterns:
            break
        elif sum_u >= 0.9:
            break

    return transform_num


def create_mol_from_pattern(pattern_nums_in_file, molecule):
    cgr_patterns = []
    destroy_all = []
    created_reactions = []

    for cgr in CGR_env_hyb:
        if cgr.meta["id"] in pattern_nums_in_file:
            cgr_patterns.append(cgr)

    for cgr_nums in range(len(cgr_patterns)):
        cgr_pattern = cgr_patterns[cgr_nums]
        cgr_to_reaction = CGRpreparer.decompose(cgr_pattern)

        prod_from_cgr = cgr_to_reaction.products
        reag_from_cgr = cgr_to_reaction.reagents

        if len(reag_from_cgr) < 3:

            react = CGRreactor()

            with RDFwrite("/home/aigul/Retro/templates/test.template") as f:
                f.write(ReactionContainer(reagents=[prod_from_cgr[0]], products=reag_from_cgr))

            with RDFread("/home/aigul/Retro/templates/test.template", is_template=True) as f:
                template = react.prepare_templates(f.read())

            searcher = react.get_template_searcher(templates=template)

            for i in searcher(molecule[0]):
                destroy = react.patcher(structure=molecule[0], patch=i.patch)

            destroy_all.append(CGRpreparer.split(destroy))

            created_reactions.append(ReactionContainer(reagents=[destroy], products=molecule))

    return destroy_all, created_reactions


def create_hash_from_pred_mols(molecule_from_prediction):
    molecule_hash_list_0 = []
    for i1 in range(len(molecule_from_prediction[0])):
        molecule_hash_list_1 = []
        for j1 in range(len(molecule_from_prediction[0][i1])):
            molecule_hash_list_1.append(molecule_from_prediction[0][i1][j1].get_signature_hash())
        molecule_hash_list_0.append(molecule_hash_list_1)
    reaction_hash_list_0 = []
    for i2 in range(len(molecule_from_prediction[1])):
        reaction_hash_list_0.append(molecule_from_prediction[1][i2].get_signature_hash())
    return molecule_hash_list_0, reaction_hash_list_0


# for target molecule
db_mols = shelve.open("/home/aigul/Retro/db_mols.txt")
target_mol_hash = target[0].get_signature_hash()
db_mols[str(target_mol_hash)] = target[0]
db_mols.close()


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
