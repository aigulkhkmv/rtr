import pickle

trans = {}
instucture = False
with open("../../Desktop/CGR.sdf") as f:
    for n, line in enumerate(f):
        if line.find(">  <env_hyb.1>") != -1:
            num = int(f.readline())
            instucture = True
        if line.find(">  <env_hyb.2>") != -1:
            instucture = False
            continue
        if line.find(">  <id>") != -1 and instucture:
            id_ = f.readline()
            if int(num) not in trans:
                trans[int(num)] = []
            trans[int(num)].append(int(id_))
        if n % 10000000 == 0:
            print(n)
with open("transformations_pycharm.pickle", "wb") as f:
    pickle.dump(trans, f)

