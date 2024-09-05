#i/bin/sh

echo "Starting the SHELL script!"

echo "There are three datasets: Heart-Disease (HD), Gene-Expr (GE) and Auto-immune Disease (AI)"

echo "There are 6 possible models: LR, SVM, DT, RF, MLP, XGB."

echo "The experiment runs for number of models in the range of (50, 100, step=10)."

echo "The experiment runs with various model agreement rates in the range of (0.5, 1.0, step=0.1)."

echo "So for each dataset, there are 6 x 6 x 6 = 216 combinations of the run."

echo "***MLP takes lot of time, run separately.***"


dataset_name=("HD")
# dataset_name = ("HD" "GE" "AI")

# dataset_combo=("HG_CL")

dataset_combo=("SW_HG" "SW_VA" "VA_HG" "VA_SW")

model_name=("SVM" "DT" "RF" "XGB" "MLP")

# model_name=("LR")

n_models=(100)
# n_models=(50 60 70 80 90 100)

agreement_rates=(.5 1.0)
# agreement_rates=(0.5 0.6 0.7 0.8 0.9 1.0)

to_optimise_model=1



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
	for set_combo in ${dataset_combo[@]}; do
		for model in ${model_name[@]}; do
			for n in ${n_models[@]}; do
				for ar in ${agreement_rates[@]}; do 
					echo "For $model, $n and agreement rate: $ar"
					path="./${n}_${model}_${ar}/"
					python3 model_agreement.py -sfg $path -dataset $dataset -dc $set_combo -m $model -n $n -ar $ar -optm $to_optimise_model 
				done
			done
		done
	done
done
