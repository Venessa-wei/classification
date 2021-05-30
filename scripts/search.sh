file_path=$(readlink -f "$(dirname "$0")")
cd $file_path/..
echo TextCNN
CUDA_VISIBLE_DEVICES=2 python run.py --model TextCNN 1> logs/TextCNN.log 2>&1

echo TextRNN
CUDA_VISIBLE_DEVICES=2 python run.py --model TextRNN 1> logs/TextRNN.log 2>&1

echo TextRNN_Att
CUDA_VISIBLE_DEVICES=2 python run.py --model TextRNN_Att 1> logs/TextRNN_Att.log 2>&1

echo FastText
CUDA_VISIBLE_DEVICES=2 python run.py --model FastText --embedding random 1> logs/FastText.log 2>&1

echo DPCNN
CUDA_VISIBLE_DEVICES=2 python run.py --model DPCNN 1> logs/DPCNN.log 2>&1

echo Transformer
CUDA_VISIBLE_DEVICES=2 python run.py --model Transformer 1> logs/Transformer.log 2>&1

