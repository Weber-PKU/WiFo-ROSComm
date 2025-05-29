【量化】
1、WiFo量化环境依赖：Cuda12.6，TensorRT>10.11.0.33（10以上应该都行）, cudnn 9.10.1.4-1 （https://developer.nvidia.com/cudnn-downloads），hdf5storage pycuda onnx等

2、量化过程（相关细节设置参考本科生给的手册）：运行build_int8_engine_rt10.py进行量化，（optional）运行run_trt_engine_rt10.py进行推理时间测试

3、转移模型：将wifo_base_int8_trt10.engine放到wifo_ros下的asset文件夹中

【节点和包编译】
4、在wifo_ros/src下创建包
catkin_create_pkg wifo_processor rospy std_msgs
，不需要管报错

5、在wifo_ros/src/wifo_processor/CMakeLists.txt中将catkin_python_setup()取消注释

Orin在工作根目录下依次运行
catkin_make（编译）

【编译完后，每次工作从此处开始】
6、在工作根目录下依次运行
chmod -R a+rwx .（可执行）

source devel/setup.bash（ros配置）

gnome-terminal --title="ROS Core" -- bash -c "source devel/setup.bash; roscore"（创建ros master）

gnome-terminal --title="RQT Topic" -- bash -c "source devel/setup.bash; rqt_topic"（监测用）

（在此之前激活对应python虚拟环境）conda activate []
gnome-terminal --title="234 WIFO" -- bash -c "source devel/setup.bash; rosrun wifo_processor wifo_.py --in-host 0.0.0.0 --in-port 12345 --out-host 0.0.0.0 --out-port 12346; exec bash"（根据需要修改IP和端口）

7、工控机运行（Python包自己配）
python3 pkg_trans.py --host 0.0.0.0 --port 12345（IP端口和上面需要保持一致）

python3 pkg_rcv.py --host 0.0.0.0 --port 12346（同理）

或者在Linux命令行中

gnome-terminal --title="1 TRANS" -- bash -c "conda activate []; python3 pkg_trans.py --host 0.0.0.0 --port 12345; exec bash"

gnome-terminal --title="5 RCV" -- bash -c "conda activate []; python3 pkg_rcv.py --host 0.0.0.0 --port 12346; exec bash"


【其他说明】
8、工作流：工控机 pkg_trans.py -> Orin wifo_.py (import trt10.py) -> 工控机 pkg_rcv.py

9、为什么用ROS写
10、后续修改变长推理的思路
11、部署到硬件上还差哪些步骤：工控机环境安装，Orin环境校正，ip






