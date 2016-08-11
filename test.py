import tensorflow as tf
import numpy as np

#this is going to be a way to test functionality of tf.cond

#define two functions which take a tensor as an input one returns a list of tensors the other returns

num_steps = 10
state_list = []

class test_rnn:

	def __init__(self):
		self.termination_tstep = tf.placeholder(tf.int32,shape = [1])
		self.tstep = tf.constant(0,shape =[1])
		self.W_fc = tf.Variable(tf.truncated_normal(shape = [5,20]))


	def decode(self,state):
		return tf.pack([tf.add(state,1)] * 4)

	def forward_step(self,state):
		return tf.add(state,5)

	def cond(self,state,tstep,termination_tstep):
		"""
		Args: Should take the loop variables as an input
		Returns: Loop Variables in the form of a list
		"""
		#as long a the tstep is less than the termination time step continue the loop
		return tf.less(tstep,termination_tstep)[0]

#next define the body of the while loop this is responsible for modifying the state at each timestep 

	def body(self,state,tstep,termination_tstep):
		return self.forward_step(state),tf.add(tstep,tf.constant(1,shape = [1])),termination_tstep	

#initialize a state_list to record the evolution of the tensor over the cycle

	def construct_graph(self,state):
		#initialize a state
		
		#initialize a state list
		#initialize a list for the loop variables 
		loop_var = [state,self.tstep,self.termination_tstep]
		#now define operation for the while loop
		r = tf.while_loop(lambda state,tstep,termination_tstep : self.cond(state,tstep,termination_tstep),lambda state,tstep,termination_tstep : self.body(state,tstep,termination_tstep), loop_var)

		sess = tf.Session()
		sess.run(tf.initialize_all_variables())

		return sess,r

	def evaluate(self,sess,r,random_index):

		output = sess.run(r[0],feed_dict = {self.termination_tstep : [random_index]})
		return output
		#generate boolean 

boolean_array = np.zeros(num_steps)
random_index = int(np.random.rand()*(num_steps))
#loss = tf.add(state_list[random_index],100)
#boolean_array[random_index] = 1
state = tf.constant(0)
#get the r 
print random_index
mylstm = test_rnn()
sess,r = mylstm.construct_graph(state)
output = mylstm.evaluate(sess,r,random_index)
print output

#for tstep in range(random_index):
	#print tstep,state_list[tstep].eval(feed_dict = {termination_tstep : random_index})

# print loss.eval(feed_dict = {boolean_input : boolean_array})