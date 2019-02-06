import pickle
from CGRtools.files import SDFwrite, RDFread
from CGRtools.containers import MoleculeContainer


with open("/home/aigul/Retro/Data_for_modeling_100.pickle", "rb") as f:
    rea_centers = pickle.load(f)
reverse = {}
for k, v in rea_centers.items():
    for i in v:
        reverse[i] = k


nums = ["1", "2", "3", "4"]
with SDFwrite("/home/aigul/Retro/training_set.sdf".format, "a") as sdf:
    for i in nums:
        for n,reaction in enumerate(RDFread("/home/aigul/Retro/parsed_part_{}.rdf".format(i))):
            if int(reaction.meta['id']) not in reverse:
                continue
            reaction.meta["rule"]=reverse[int(reaction.meta['id'])]
            for x in reaction.products:
                if x.number_of_nodes()>5:
                    sdf.write(MoleculeContainer(data=x,meta=reaction.meta))