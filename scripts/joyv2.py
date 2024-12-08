import rospy
from sensor_msgs.msg import Joy
import matplotlib.pyplot as plt

def joy_callback(data):
    left_x = data.axes[0]
    left_y = data.axes[1]
    right_x = data.axes[2]
    right_y = data.axes[3]
    
    # Update plot
    left_stick.set_xdata(left_x)
    left_stick.set_ydata(left_y)
    right_stick.set_xdata(right_x)
    right_stick.set_ydata(right_y)
    plt.draw()

rospy.init_node('joystick_visualizer')
rospy.Subscriber("/joy", Joy, joy_callback)

fig, (ax1, ax2) = plt.subplots(1, 2)
left_stick, = ax1.plot(0, 0, 'ro')
right_stick, = ax2.plot(0, 0, 'bo')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)

plt.show()
rospy.spin()
