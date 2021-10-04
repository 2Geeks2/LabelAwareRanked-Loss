# LabelAwareRanked-Loss

In this repository, we implement different kinds of losses which are mainly Triplet loss, Multiclass-N-pair loss, Constellation loss, and LabelAwareRanked loss. We use the same smart batch structure to test. Batch is selected by using `BalancedBatchSampler`

## Random Data Experiment
The following image shows randomly generated datapoints on a unit circle that become clustered and ranked in uniform angles after applying and optimizing the LAR loss.
<img src="figures/random_data_experiment.svg" width="600">

Despite the clustering, we can also see that the loss converges to its minimum which is achieved for uniform angles and ranking between different labels.
<img src="figures/random_data_experiment_loss.svg" width="600">

The experiment on a randomly generated dataset shows that the LAR loss creates ranked embeddings in uniform angles when it is close to the optimal solution. This experiment can be executed by the following command:

- "num_classes" ---- number of classes of the dataset (int)
- "num data" ---- number of datapoints  (int)
- "num_iter" ---- number of iterations (int)
- "plot" --- "True" or "False" for the plots

<pre><code>python gradient_descent_rnd_data.py --num_classes 7 --num_data 1000 --num_iter 1000 --plot True
</code></pre>
For MNIST the experiment can be executed by the following script. There are two parameters you can choose.
## MNIST Data Experiment
1. Loss: 
   - "triplet" ---- Triplet loss
   - "npair" --- Multiclass-N-pair loss
   - "constellation" --- Constellation loss
   - "lar" --- LabelAwareRanked loss
2. Number of epochs: choose a integer

Run the code in command line like following:

<pre><code>python main.py --loss lar --num_epochs 10
</code></pre>


<img src="figures/label-aware-ranked_loss.svg" width="600">
