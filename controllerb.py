"""
This module contains the transfer function which is responsible for determining the muscle movements of the myoarm
"""
from sensor_msgs.msg import JointState

from std_msgs.msg import Float64
@nrp.MapRobotPublisher("joint1", Topic("/robotb/arm_1_joint/cmd_pos", std_msgs.msg.Float64)) 
@nrp.MapRobotPublisher("rewardb_ros", Topic("/rewardb", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("links", Topic("/gazebo/link_states", gazebo_msgs.msg.LinkStates)) 
@nrp.MapRobotPublisher("joint4", Topic("/robotb/arm_4_joint/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint2", Topic("/robotb/arm_2_joint/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint5", Topic("/robotb/arm_5_joint/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint3", Topic("/robotb/arm_3_joint/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("joints", Topic("/jointState", sensor_msgs.msg.JointState))
@nrp.MapVariable("agent", initial_value=None)
@nrp.MapVariable("session2", initial_value=None)
@nrp.MapVariable("graph2", initial_value=None)
#actuators


@nrp.Neuron2Robot()
def controllerb (t, joints, links, joint2, joint3, joint4, joint5, rewardb_ros, joint1, agent, graph2, session2):
   
    if agent.value is None:    
        # import keras-rl in NRP through virtual env
        import site, os
        site.addsitedir(os.path.expanduser('~/.opt/tensorflow_venv/lib/python2.7/site-packages'))
        from keras.models import Model, Sequential
        from keras.layers import Dense, Activation, Flatten, Input, concatenate
        from keras.optimizers import Adam, RMSprop
        from rl.agents import DDPGAgent
        from rl.memory import SequentialMemory
        from rl.random import OrnsteinUhlenbeckProcess
        from keras import backend as K 
      
        from tensorflow import Session, Graph
        K.clear_session()

        obs_shape = (6,)

        nb_actions = 5


        
    # create the nets for rl agent
     # actor net
        
        graph2.value = Graph()
        with graph2.value.as_default():
                session2.value = Session()
                with session2.value.as_default():
                            
                    actor = Sequential()
                    actor.add(Flatten(input_shape=(1,) + obs_shape))
                    actor.add(Dense(32))
                    actor.add(Activation('relu'))
                    actor.add(Dense(32))
                    actor.add(Activation('relu'))
                    actor.add(Dense(32))
                    actor.add(Activation('relu'))
                    actor.add(Dense(nb_actions))
                    actor.add(Activation('sigmoid'))
                    clientLogger.info('actor net init')

                    # critic net
                    action_input = Input(shape=(nb_actions,), name='action_input')
                    observation_input = Input(shape=(1,) + obs_shape, name='observation_input')
                    flattened_observation = Flatten()(observation_input)
                    x = concatenate([action_input, flattened_observation])
                    x = Dense(64)(x)
                    x = Activation('relu')(x)
                    x = Dense(64)(x)
                    x = Activation('relu')(x)
                    x = Dense(64)(x)
                    x = Activation('relu')(x)
                    x = Dense(1)(x)
                    x = Activation('linear')(x)
                    critic = Model(inputs=[action_input, observation_input], outputs=x)
                    clientLogger.info('critic net init')
        
                    # instanstiate rl agent
                    memory = SequentialMemory(limit=1000, window_length=1)
                    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=nb_actions)
                    agent.value = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=10,   nb_steps_warmup_actor=10, random_process=random_process, gamma=.99, batch_size=5, target_model_update=1e-3, delta_clip=1.)
                    agent.value.training = True
                    clientLogger.info('rl agent init')
        
                    PATH = '/home/user/WORK/NRP/NRP-local/Experiments/bf_manipulation_demo/ddpg_weights.h5'
                    if os.path.isfile(PATH):
                            print('loading weights')
                            agent.load_weights(PATH)
                            clientLogger.info('weights loaded')
        
                    agent.value.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
                    clientLogger.info('agent compiled - ready to use')

                
               
#### run steps

    #graph1.value = Graph()
    with graph2.value.as_default():
     #       session1.value = Session()
            with session2.value.as_default():
                            
                import math
                import numpy as np
     
                angle_lower = links.value.pose[5].position.x
                angle_vel_lower = links.value.pose[7].position.x
                angle_upper = links.value.pose[9].position.x
                angle_vel_upper = links.value.pose[12].position.x
   #  clientLogger.info('humerus_angle ', links.value.pose[15].position.y)  
      #  clientLogger.info('humerus_ang_vel ', angle_vel_lower)
      #  clientLogger.info('radius_angle ', angle_upper)
      #  clientLogger.info('radius_ang_vel ', angle_vel_lower)

                observation = np.array([math.cos(angle_lower), math.sin(angle_lower),
                                angle_vel_lower, math.cos(angle_upper),
                                math.sin(angle_upper), angle_vel_upper])

                # get movement action from agent and publish to robot
                action = agent.value.forward(observation)
                clientLogger.info('agent stepped foward')
    
                # move robot
                joint1.send_message(std_msgs.msg.Float64(action[0]))    
                joint2.send_message(std_msgs.msg.Float64(-action[1]))
                joint3.send_message(std_msgs.msg.Float64(action[2]))
                joint4.send_message(std_msgs.msg.Float64(action[3]))
                joint5.send_message(std_msgs.msg.Float64(action[4]))

    
        
                import math
                reward = \
                math.sqrt(math.pow((links.value.pose[57].position.x - links.value.pose[4].position.x),2) + \
                math.pow((links.value.pose[57].position.x - links.value.pose[4].position.x),2) + \
                math.pow((links.value.pose[57].position.x - links.value.pose[4].position.x),2))
        
 
                clientLogger.info('REWARD IS:',  reward)
                rewardb_ros.send_message(reward)
        ## reward x müsste minimiert für runter!
    #-(angle_lower**2 + 0.1*angle_vel_lower**2 +
              #     angle_upper**2 + 0.1*angle_vel_upper**2 +
              #     0.001*np.sum(np.power(action, 2)))

        #learn from the reward
                agent.value.backward(reward)
                clientLogger.info('agent stepped backward')
                agent.value.step = agent.value.step + 1


                if agent.value.step%20 == 0:
                    clientLogger.info('saving weights')
                    PATH = '/home/user/Desktop/keras_learning_weights/ddpg_weights_b.h5'
                    agent.value.save_weights(PATH, overwrite=True)
        
                clientLogger.info('-------one step done')


            
