# Transferable Visuomotor Representations #

We seek to investigate the computational underpinnings that facilitate imitation learning. We will do this by first constructing a toy world consisting of robotic arms which are restricted to move in 2-Dimensions over a grid. The end of each robotic arm will be able to mark its position on the grid. Furthermore, each robotic arm may consist of two or more links. A motor at the intersection of two links will be able to determine the angle between the pair. By changing the angles between each pair of links we will be able to alter the state of the arm and hence the position of its end effector. This will allow us to draw shapes on the grid!

### Task 1 ###

As a Baseline we seek to simply reconstruct the output images via an Autoencoder. Two architectures are used specifically Image_AutoEncoder_ver1.py and Image_AutoEncoder_ver2.py.

### Task 2 ###

We then seek to demonstrate that an arm should be able to replicate the output image produced by another arm by simply learning the transformation between its state and the latter arm's state. We initially restrict our attention to two two link arms with different link lengths. We attempt to learn the transformation via a variable sequence to sequence mapping implemented in variable_seq2seq.py.

### Experiment 1 ###

Possible to form a hidden representation for rectangles and triangles and then reconstruct them.

### Experiment 2 ###

Demonstrate that one agent may learn a sequence of desired states to achieve a goal by accessing a second agent's state sequence required to perform the same goal.

### Experiment 3 ###

Demonstrate that it is possible for an agent to infer the required state sequence by observing a sequence of the goal outputs.
