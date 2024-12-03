import os
import datetime
stamp = datetime.datetime.now().strftime("%H%M%S")
# override manually to call train/test
man_stamp = 141547
collect_prefix = os.environ['HOME'] + f'/imitation_learning_ros/src/imitation_learning/data/{stamp}/'
collect_prefix_post = os.environ['HOME'] + f'/imitation_learning_ros/src/imitation_learning/{man_stamp}/'   
file_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '/'

prefix = os.environ['HOME'] + f'/imitation_learning_ros/src/imitation_learning/data/{stamp}/'