
random: experiment
	python2 random_agent.py

mlp: experiment
	python2 mlp_agent.py

experiment:
	rl_glue &
	java -jar rl-library/products/MountainCar.jar &
	python2 experiment.py &

