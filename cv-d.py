from subprocess import Popen, PIPE
from tqdm import tqdm
from multiprocessing import Pool

def func(param):
    k, C, d, i = param
    print(f"worker ({k}, {C}, {d}, {i})")
    model=f"./cv-d/model/model.d{d}.k{k}.fold{i}"
    cmd=f"./libsvm-3.24/svm-train -t 0 -c {C} "\
    f"train.scale.transformed{d}.{i} {model} "\
    f"> ./cv-d/train/train.d{d}.k{k}.fold{i}.txt"
    print(cmd)
    Popen(cmd, shell=True).wait()
    cmd=f"./libsvm-3.24/svm-predict val.scale.transformed{d}.{i} {model} /dev/null"
    print(cmd)
    proc=Popen(cmd, shell=True, stdout=PIPE, universal_newlines=True)
    return proc.communicate()[0]

p = Pool(6)
nfold = 10
#     for k in tqdm(range(16, -11, -1)):
for d in range(3, 5):
    st = -20
    ed = -10
    # st = [-25, -30, -40, -40][d-1]
    # ed = -10
    # st, ed = -2, 10
    for k in tqdm(range(st, ed)):
# for k in tqdm(range(-10, 17)):
# for k in tqdm(range(17, 30)):
    # for d in tqdm(range(1, 5)):
        if k%5==0: continue
        C = 2**k
        print(f"({d},{C})")
        params = list(zip([k]*nfold, [C]*nfold, [d]*nfold, range(nfold)))
        print(params)
        result = "".join(p.imap(func, params))
        with open(f'./cv-d/result/result.d{d}.k{k}', 'w') as fout:
            fout.write(result)