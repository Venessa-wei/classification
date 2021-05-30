import itertools
import os


def main():
    models = ['RF', 'LR', 'DT', 'KNN', 'SVM', '', 'GDBT']
    lrs = [0.0009, 0.001, 0.002, 0.003, 0.005, 0.008]
    ks = range(1, 50)
    DEVICE = 3
    _model = models[3]
    if not os.path.exists(f'../{_model}logs'):
        os.makedirs(f'../{_model}logs')
    with open('run.sh', 'w') as out:
        # 文件路径
        out.write('file_path=$(readlink -f "$(dirname "$0")")\ncd $file_path/..\n')
        for k in ks:
            out.write(f"echo KNN-{k}\n")
            out.write(f"python -u main.py --model KNN --k {k} --gpu-id 0 --k-fold ")
            out.write(f"1> {_model}logs/KNN-{k}.log 2>&1")
            out.write("\n\n")
        # for lr in lrs:
        #     out.write(f"echo NN++ lr={lr}\n")
        #     out.write(
        #         f"python -u main.py --model NN++ --gpu-id 0 --batch-size 10000 --hidden-size 32  --max-patience 600  --learning-rate {lr} --k-fold ")
        #     out.write(f"1> logs/NN++_{lr}.log 2>&1")
        #     out.write("\n\n")
        # for md in models:
        #     out.write(f"echo {md}++\n")
        #     out.write(f"python -u main.py --model {md}++ --gpu-id 0 --k-fold ")
        #     out.write(f"1> logs/{md}++.log 2>&1")
        #     out.write("\n\n")
        # for md in models:
        #     out.write(f"echo {md}\n")
        #     out.write(f"python -u main.py --model {md} --gpu-id 0 --k-fold ")
        #     out.write(f"1> logs/{md}.log 2>&1")
        #     out.write("\n\n")


if __name__ == '__main__':
    main()
