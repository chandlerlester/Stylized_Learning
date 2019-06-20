# Stylized Steady State Learning Rules 

This is a repository of my third year field paper code. 
In my field paper, 
I examined what happens when an agent receives new information and is able to update their estimate of the economy's steady state in a simple Ramsey model. 

To accomplish this task I implement two different types of stylized learning rules. 

### Observing Noisy Parameter Values 

For the first exercise I allowed the agent to observe noisy parameter estimates and then update their own estimate of the parameters using this information. 

Code for this learning exercise can be found in: 

- Stylized_Rule_Capital
- Stylized_Rule_Prod 

### Observing Data points 

This second exercise allows the agent to observe actual points in a continuous stochastic process. 
The agent then uses these points to generate an estimate of the true parameters values. 
Next, the agent uses recursive least squares (RLS) to update this estimates as they observe more data. 

Code for this learning exercise can be found in: 

- Updating_const_gain
- Updating_const_gain_RLS2 
- Updating_decreasing_gain
- Updating_decreasing_gain_RLS2 

The folders labeled "RLS2" use a version of RLS that uses multiple data points for one calculation. 
Folders without this label run a version of the code that uses a more traditional RLS formula; 
here the agents run RLS on each data point even though they observe multiple data points in one time interval. 

### My Paper 
For a full explanation of these exercises see my [field paper](https://chandlerlester.com/images/Field-Paper-2019-06-20.pdf). 
