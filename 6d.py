import numpy as np
from tqdm import tqdm
def Ker(x1, x2, d):
    return np.dot(x1,x2)**d
def readData(path):
    Xs = []
    ys = []
    with open(path) as fin:
        for line in fin.readlines():
            a=line.strip().split(' ')
            X = np.zeros(10)
            for j in range(1, len(a)):
                idx,val = a[j].split(':')
                X[int(idx)-1] = float(val)
            Xs.append(X)
            ys.append(int(a[0]))
    Xs=np.array(Xs)
    ys=np.array(ys)
    return Xs, ys
def dumpData(path, Xs, ys):
    with open(path, 'w') as fout:
        for X,y in zip(Xs,ys):
            s = [f"{i+1}:{x}" for i,x in enumerate(X)]
            s = f"{y} " + " ".join(s)
            fout.write(s+"\n")
def transform(Xs, baseXs, ys, d):
    mt,n = Xs.shape
    mb = baseXs.shape[0]
    K = np.matmul(Xs, np.transpose(baseXs)) ** d
    print(K.shape)
    nXs = np.matmul(K, np.diag(2*ys-1))
    for i in range(mt):
        for j in np.random.choice(mb, 10):
#         X = [(2*ys[j]-1)*Ker(Xs[i],Xs[j],d) for j in range(m)]
            x = (2*ys[j]-1)*Ker(Xs[i],baseXs[j],d)
            assert ((x - nXs[i,j])<1e-6).all(), (x, nXs[i,j])
    print('check passed')
    return np.array(nXs)

nfold = 10
for i in tqdm(range(nfold+1)):
    suf       = f".{i}" if i < nfold else "" # one fold or the original data
    test_name = 'val' if i < nfold else "test" 
    Xs_train, ys_train = readData(f'./train.scale{suf}')
    Xs_test, ys_test = readData(f'./{test_name}.scale{suf}')
    for d in range(1, 5):
        nXs_train = transform(Xs_train, Xs_train, ys_train, d)
        nXs_test = transform(Xs_test, Xs_train, ys_train, d)
        assert len(nXs_train) == len(ys_train)
        assert len(nXs_test) == len(ys_test)
        dumpData(f'./train.scale.transformed{d}{suf}', nXs_train, ys_train)
        dumpData(f'./{test_name}.scale.transformed{d}{suf}', nXs_test, ys_test)
