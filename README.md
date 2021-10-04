# LabelAwareRanked-Loss
## Code execution
The experiment for a random dataset with gradient descent can be executed by

<pre><code>python gradient_descent_rnd_data.py --num_classes 7 --num_data 1000 --num_iter 1000 --plot True
</code></pre>

For MNIST the experiment can be executed by the following script. There is also the possibility to run the experiment weith different losses
<pre><code>python main.py --lar
</code></pre>

![lar loss](figures/label-aware-ranked_loss.svg)
