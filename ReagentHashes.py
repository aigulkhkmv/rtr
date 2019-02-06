from CGRtools.files import RDFread, RDFwrite, SDFread, SDFwrite
import joblib
from joblib import Parallel, delayed
import pickle

with SDFread("/home/aigul/Desktop/mol_db_stand.sdf") as f3:
    reagents = f3.read()

def arom(reagent):
    return standardizer.transform([reagent]).as_matrix()[0].get_signature_hash()

stand_reag_hashes = Parallel(n_jobs=4, verbose=1)(delayed(arom)(reagent) for reagent in reagents)

with open("/home/aigul/Retro/reagents/molprotbb_std.pickle", "wb") as f:
    pickle.dump(stand_reag_hashes, f)


