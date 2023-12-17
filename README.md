# activation_analysis
## 1. Introduction
### 1.1 What we are doing?

We are trying to evaluate the quality of a model's prediction by another neural network(noted as meta-network). This task can be applied in scenarios that require real-time assessment of a model's output, such as autonomous driving. In such contexts, it's crucial to evaluate the reliability of the model's instantaneous output to make appropriate decisions.

### 1.2 Two tasks 
We split the problem into two tasks, one is the regression task and the other is classification task.
#### 1.2.1 Regression task
One way to measure the quality of the model's output is the robustness. If the model's prediction is robust against strong random noise, we can say such a model is 'confident' on such a prediction. In fact, a good machine learning model should make consist prediction when facing noise.

Therefore, we use gaussian random noise to disturb the images and calculate the ratio of times when the model does not change its prediction on the noised images. Detail steps are shown in part 2.2. The higher the ratio, the more confident the model is on related images. 

In regression task, we try to train a meta-network to predict the confidence ratios of the given models and given images.
### 1.2.2 Classification task
Another way to measure the quality of the prediction from the tested model is correctness. The wrong prediction absolutely is not so good as the right prediction. In this task, we do not need to generate labels like we do in the regression task. We only need to train a meta-network to prediction the correctness of the tested model's output.

## 2. Data Preparation

### 2.1 Composition of Inputs

To achieve the goal, the input of evaluating networks should be consist of the following two components:

1. Data flow (Required, directly shows the transformations of the input data between each layers)
2. Information of the network being tested (Optional)

If we do not include the data flow in the input, the task will turn to evaluate a performance of a given neural network, since a fixed network always has difference confidences for different input samples.

The structure and weight of the network completely determine the transformation process of an input sample in the network, so the information of the network may not be necessary from the information theory point of view. 

However, we cannot guarantee that our evaluating newtork will fully use information from the data flow. Therefore, maybe we should also consider the information of the testing network as a part of our input. 

### 2.2 Design of the label in regression task

Since our job is to predict the confidence level of a prediction from the network, we need a metric to evaluate it so that the result of the metric given the specific input can be used as the lable for our meta-network training process. To simplify the discussion, **we only consider to evaluation classification models (such as LeNet) in this project**. 

1. Using the output of the softmax layer might not be a good idea. 

   1. Because there is no strict guarantee that the softmax output represents the sample’s distribution about the classes well. we can only say there is a good chance that the class related to the maximum of the softmax output is the true label of the sample according to the training schema.
   2. Also, if we use the softmax output as the label indicator, we will not need to train a model to evaluate the confidence level of a prediction. Instead, we can just focus on the softmax output of the sample.

2. An alternetive way is to use sampling method to approximate the robustness of a prediction.

   1. Add a random perturbation to a sample to generate several new samlpes.
   2. Get predictions of those new samples.
   3. Calculate how many percent of the predictions are different with the original one or the KL-divergence of the softmax outputs.

   The intuition is that if the network is confident about its prediction, is should be robust and not influenced by little perturbation.

### 2.3 Examples

In this project, we use `LeNet5` as the model being tested, `MNIST-10` as the input data for those models. More specifically:

1. We train thousands of `LeNet5`s with different hyper-parameter settings.
2. For each model, we select 200 samples from the test set of `MNIST-10`.
   - Each digit has 20 selected samples
   - 10 for right predictions and 10 for wrong predictions
   - some models (too bad or too good) might not allow us to select enough samples
3. For each model-sample pair (ideally, a model will have 200 pairs), we add random noise to the image sample respectively, then calculate how many percent of the predictions are not changed by the noise. We use the ratio as the lable for our meta-network training. 

## 3. How to Re-prodcue our Result
### 3.1 Train thousands of LeNet-5 models with different hyper-parameter settings.
Run the following command in the terminal:
```shell
python TrainLenet.py
```
This command will generate more than 4,000 LeNets with different hyperparameters and accuarcy >= $60\%$.

### 3.2 Sample images for each trained LeNet-5 model and generate the `ratio` label for each model-image pair.
Run the following command in the terminal:
```shell
python DataPreparation.py
```
This script will sample about 200 digit images for each model in directory `models` into the directory `model_sample`. Each model will be assigned:
- 100 samples classified correectly, each class with 10 samples.
- 100 samples classified incorrectly, each class with 10 samples.

Then, run the following command:
```shell
python gen_pair.py
```
This will generate files in the following directories:
1. `pair_9_10/`: Containing models' weight, sampled images, data flow during the inference, ratio (represente the confidence of the prediction), type(correct prediction or wrong prediction).
2. `pair_9_10_npy/`: Containing the last four layers' output of the model given the images: `x_maxpool2`, `x_fc1`, `x_fc2`, `x_fc3`, ratio (represente the confidence of the prediction), type(correct prediction or wrong prediction).
3. `pair_9_npy`: Containing 1/10 data in `pair_9_10_npy/`.
### 3.3 Train meta-networks to predict the confidence or correctness of the prediction from given LeNet-5 and images.
We provide three kinds of Meta-Networks:
1. RNN based
2. LSTM based
3. TransformerEncoder based

The following .sh files will train those models with random seed as 1,2,3,4,5:
1. `train_rnn.sh`
2. `train_lstm.sh`
3. `train_transformer.sh`

The training logs can be seen in `log/`. The training result can be seen in `results/`. 

Overall, 
1. The best reg model is lstm based. It model can achieve $r^2>0.77$.
2. The best classification model is rnn based. It can achieve accuracy $>0.91$.