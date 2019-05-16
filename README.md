g++ my_transe.cpp -o my_transe -pthread -O3 -std=c++11 -march=native

./my_transe

source activate py3

python accuracy.py
