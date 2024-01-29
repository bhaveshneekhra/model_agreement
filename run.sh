#i/bin/sh

echo "Starting the SHELL script!"

echo "There are two datasets: Heart-Disease (HD) and Gene-Expr (GE)"

echo "There are 6 possible models: LR, SVM, DT, RF, MLP, XGB."

echo "The experiment runs for number of models in the range of (50, 100, step=10)."

echo "The experiment runs with various model agreement rates in the range of (0.5, 1.0, step=0.1)."

echo "So for each dataset, there are 6 x 6 x 6 = 216 combinations of the run."

echo "***MLP takes lot of time, run separately.***"


dataset_name=("HD")
# dataset_name = ("HD" "GE")

model_name=("LR" "DT" "RF" "MLP" "XGB")

# model_name=("SVM")

# n_models=(100)
n_models=(50 60 70 80 90 100)

agreement_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
# agreement_rates=(1.0)
# 0.5 0.6 0.7 0.8 0.9 1.0)

# echo "model names:"
# for model in ${model_name[@]}; do 
# 	echo $model;
# done

# echo "Number of models:"
# for n in ${n_models[@]}; do 
# 	echo $n;
# done

# echo "Agreement rates:"
# for ar in ${agreement_rates[@]}; do 
# 	echo $ar;
# done

for dataset in ${dataset_name[@]};do
	for model in ${model_name[@]}; do
		for n in ${n_models[@]}; do
			for ar in ${agreement_rates[@]}; do 
				echo "For $model, $n and agreement rate: $ar"
				path="./${n}_${model}_${ar}/"
				python3 model_agreement.py -sfg $path -dataset $dataset -m $model -n $n -ar $ar
			done
		done
	done
done
