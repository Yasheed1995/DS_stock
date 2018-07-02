python run.py LinR

for model in LogR SVM D-Tree RF NN KMeans Bayes
do
    python run.py $model 11
done
