from subprocess import Popen, PIPE
from tqdm import tqdm
#     for k in tqdm(range(16, -11, -1)):
for k in tqdm(range(-10, 17)):
# for k in tqdm(range(17, 30)):
    for d in tqdm(range(1, 5)):
        C = 2**k
        print(f"({d},{C})")
        result = ""
        for i in range(10):
            model=f"./cv/model/model.d{d}.k{k}.fold{i}"
            cmd=f"./libsvm-3.24/svm-train -t 1 -c {C} -d {d} "\
            f"train.scale.{i} {model} "\
            f"> ./cv/train/train.d{d}.k{k}.fold{i}.txt"
            print(cmd)
            Popen(cmd, shell=True).wait()
            cmd=f"./libsvm-3.24/svm-predict val.scale.{i} {model} /dev/null"
            print(cmd)
            proc=Popen(cmd, shell=True, stdout=PIPE, universal_newlines=True)
            result+=proc.communicate()[0]
        with open(f'./cv/result/result.d{d}.k{k}', 'w') as fout:
            fout.write(result)