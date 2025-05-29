#!/usr/bin/env python3
# rviz_pub.py
# Subscribe to /wifo/iq_visual, publish I/Q as discrete points on z=0 plane,
# fitting into a ~10 m × 10 m canvas, and log info after each publish.

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from wifo_processor.msg import VizPack  # custom message

def iq_callback(msg: VizPack):
    seq = msg.seq
    stamp = msg.header.stamp.to_sec()
    data = msg.data_IQ
    N2 = len(data)
    N = N2 // 2
    if N == 0:
        rospy.logwarn("[rviz_pub] Received empty data")
        return

    # split I and Q
    I_list = data[0::2]
    Q_list = data[1::2]

    # canvas dimensions (meters)
    canvas_width = 10.0
    canvas_height = 10.0

    # spacing and scaling
    x_spacing = canvas_width / (N - 1) if N > 1 else 0.0
    max_abs = max(
        max(abs(v) for v in I_list),
        max(abs(v) for v in Q_list),
        1
    )
    y_scale = (canvas_height / 2.0) / max_abs  # map values into [-h/2, +h/2]

    # prepare I-channel POINTS marker
    m_i = Marker()
    m_i.header = msg.header
    m_i.ns = 'wifo_wave'
    m_i.id = 0
    m_i.type = Marker.POINTS
    m_i.action = Marker.ADD
    # set a valid orientation quaternion
    m_i.pose.orientation.w = 1.0
    m_i.scale.x = 0.02  # point width
    m_i.scale.y = 0.02  # point height
    m_i.color.r = 1.0
    m_i.color.a = 1.0

    # prepare Q-channel POINTS marker
    m_q = Marker()
    m_q.header = msg.header
    m_q.ns = 'wifo_wave'
    m_q.id = 1
    m_q.type = Marker.POINTS
    m_q.action = Marker.ADD
    m_q.pose.orientation.w = 1.0
    m_q.scale.x = 0.02
    m_q.scale.y = 0.02
    m_q.color.b = 1.0
    m_q.color.a = 1.0

    # optional: text marker showing seq and timestamp
    m_text = Marker()
    m_text.header = msg.header
    m_text.ns = 'wifo_info'
    m_text.id = 2
    m_text.type = Marker.TEXT_VIEW_FACING
    m_text.action = Marker.ADD
    m_text.pose.position.x = 0.0
    m_text.pose.position.y = 0.0
    m_text.pose.position.z = canvas_height / 2.0 + 1.0  # above data plane
    m_text.scale.z = 0.5  # text height
    m_text.color.g = 1.0
    m_text.color.a = 1.0
    m_text.text = f"Seq: {seq}, Time: {stamp:.3f}"

    # fill point lists
    for i in range(N):
        x = i * x_spacing
        y_i = I_list[i] * y_scale
        y_q = Q_list[i] * y_scale
        m_i.points.append(Point(x=x,    y=y_i, z=0.0))
        m_q.points.append(Point(x=x,    y=y_q, z=0.0))

    # publish as MarkerArray
    marker_array = MarkerArray()
    marker_array.markers.append(m_i)
    marker_array.markers.append(m_q)
    marker_array.markers.append(m_text)
    pub.publish(marker_array)

    # log after publishing
    rospy.loginfo(
        f"[rviz_pub] seq={seq}, stamp={stamp:.3f}, "
        f"points={N}, canvas=({canvas_width}×{canvas_height} m)"
    )

if __name__ == '__main__':
    rospy.init_node('wifo_rviz_pub', anonymous=False)
    pub = rospy.Publisher(
        '/visualization_marker_array',
        MarkerArray,
        queue_size=1
    )
    rospy.Subscriber(
        '/wifo/iq_visual',
        VizPack,
        iq_callback
    )
    rospy.loginfo("[rviz_pub] Node started, listening on /wifo/iq_visual")
    rospy.spin()
