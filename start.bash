cd ~/wifo_ros || exit

chmod -R a+rwx .

conda activate ros

catkin_make

source devel/setup.bash

gnome-terminal --title="ROS Core" -- bash -c "source devel/setup.bash; roscore"

# gnome-terminal --title="RQT Graph" -- bash -c "source devel/setup.bash; rqt_graph"

gnome-terminal --title="RQT Topic" -- bash -c "source devel/setup.bash; rqt_topic"

gnome-terminal --title="234 WIFO" -- bash -c "source devel/setup.bash; rosrun wifo_processor wifo_.py --in-host 0.0.0.0 --in-port 12345 --out-host 0.0.0.0 --out-port 12346; exec bash"

# LISTEN_HOST = rospy.get_param('~listen_host', '')  # 默认监听所有接口 LISTEN_PORT = rospy.get_param('~listen_port', 8000)  # 默认端口8000

# python3 pkg_trans.py --host 0.0.0.0 --port 34791
# python3 pkg_rcv.py --host 0.0.0.0 --port 34791

sleep 2

gnome-terminal --title="1 TRANS" -- bash -c "conda activate ros; python3 pkg_trans.py --host 0.0.0.0 --port 12345; exec bash"

gnome-terminal --title="5 RCV" -- bash -c "conda activate ros; python3 pkg_rcv.py --host 0.0.0.0 --port 12346; exec bash"