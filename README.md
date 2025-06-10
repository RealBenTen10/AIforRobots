# Exercises for AI for Robots 2025

## Task 7.2

Clone the whole repository or just the [JupyterNotebook](Task7/Task7.2.ipynb).
If you haven't used JupyterNotebook before, make sure to install all necessary packages and extensions for your IDE.
JupyterNotebook runs best inside a virtual environment on PyCharm (according to my experience).
When everything is installed, simply click on "Run all" (looks like two play buttons) and the code should be executed.

The notebook features explanation of the code for each task a) - c) and e.g. for c) also for i) till iii). The notebook 
should already show the last used parameters and it's plots but be free to play around ^^

## Task 6.1

Clone the whole repository or just the [JupyterNotebook](Task6/Task6.ipynb) and the [data](Task6/data.txt)
If you haven't used JupyterNotebook before, make sure to install all necessary packages and extensions for your IDE.
JupyterNotebook runs best inside a virtual environment on PyCharm (according to my experience).
When everything is installed, simply click on "Run all" (looks like two play buttons) and the code should be executed.

We decided to use the following approach to solve the task:
- use the [python script](Task6/task6.py) to save the ir_intensity into a txt file (we collect 10 seconds worth)
- read the data into the notebook and ensure during the parsing that the data was collected correctly
- we store the data into a dictionary with the second as the key and the values in an array
- next we print a bar plot with the mean of each key (second)
- furthermore we plot the probabilities as predicted by beam-based sensor model 
- the result is not really similar to the lecture slides


## Task 5.2

Clone the whole repository or just the [JupyterNotebook](Task5/Task5.2.ipynb).
If you haven't used JupyterNotebook before, make sure to install all necessary packages and extensions for your IDE.
JupyterNotebook runs best inside a virtual environment on PyCharm (according to my experience).
When everything is installed, simply click on "Run all" (looks like two play buttons) and the code should be executed.

We decided to use the following approach to solve the task:
- explore each arm 33 times (99 in total)
- choose arm with the lowest standard deviation so far
- play arm and predict mean of previous observations


## Task 3.2
Clone the whole repository or just the [JupyterNotebook](Task3/task3.ipynb).
If you haven't used JupyterNotebook before, make sure to install all necessary packages and extensions for your IDE.
JupyterNotebook runs best inside a virtual environment on PyCharm (according to my experience).
When everything is installed, simply click on "Run all" (looks like two play buttons) and the code should be executed.

### Further information

Since we are only interested in the first 3 movements (afterwards it will always repeat the previous two steps - i.e. left, left ; left, right ; right, right) the number of steps is set to only 8. You are free to change this.

We set the initial start of the robot to be random as well - seemed feasible to us.


