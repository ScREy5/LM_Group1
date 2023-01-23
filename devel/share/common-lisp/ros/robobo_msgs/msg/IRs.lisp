; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-msg)


;//! \htmlinclude IRs.msg.html

(cl:defclass <IRs> (roslisp-msg-protocol:ros-message)
  ((FrontC
    :reader FrontC
    :initarg :FrontC
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range))
   (FrontR
    :reader FrontR
    :initarg :FrontR
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range))
   (FrontRR
    :reader FrontRR
    :initarg :FrontRR
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range))
   (FrontL
    :reader FrontL
    :initarg :FrontL
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range))
   (FrontLL
    :reader FrontLL
    :initarg :FrontLL
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range))
   (BackC
    :reader BackC
    :initarg :BackC
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range))
   (BackR
    :reader BackR
    :initarg :BackR
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range))
   (BackL
    :reader BackL
    :initarg :BackL
    :type sensor_msgs-msg:Range
    :initform (cl:make-instance 'sensor_msgs-msg:Range)))
)

(cl:defclass IRs (<IRs>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <IRs>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'IRs)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-msg:<IRs> is deprecated: use robobo_msgs-msg:IRs instead.")))

(cl:ensure-generic-function 'FrontC-val :lambda-list '(m))
(cl:defmethod FrontC-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:FrontC-val is deprecated.  Use robobo_msgs-msg:FrontC instead.")
  (FrontC m))

(cl:ensure-generic-function 'FrontR-val :lambda-list '(m))
(cl:defmethod FrontR-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:FrontR-val is deprecated.  Use robobo_msgs-msg:FrontR instead.")
  (FrontR m))

(cl:ensure-generic-function 'FrontRR-val :lambda-list '(m))
(cl:defmethod FrontRR-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:FrontRR-val is deprecated.  Use robobo_msgs-msg:FrontRR instead.")
  (FrontRR m))

(cl:ensure-generic-function 'FrontL-val :lambda-list '(m))
(cl:defmethod FrontL-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:FrontL-val is deprecated.  Use robobo_msgs-msg:FrontL instead.")
  (FrontL m))

(cl:ensure-generic-function 'FrontLL-val :lambda-list '(m))
(cl:defmethod FrontLL-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:FrontLL-val is deprecated.  Use robobo_msgs-msg:FrontLL instead.")
  (FrontLL m))

(cl:ensure-generic-function 'BackC-val :lambda-list '(m))
(cl:defmethod BackC-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:BackC-val is deprecated.  Use robobo_msgs-msg:BackC instead.")
  (BackC m))

(cl:ensure-generic-function 'BackR-val :lambda-list '(m))
(cl:defmethod BackR-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:BackR-val is deprecated.  Use robobo_msgs-msg:BackR instead.")
  (BackR m))

(cl:ensure-generic-function 'BackL-val :lambda-list '(m))
(cl:defmethod BackL-val ((m <IRs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:BackL-val is deprecated.  Use robobo_msgs-msg:BackL instead.")
  (BackL m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <IRs>) ostream)
  "Serializes a message object of type '<IRs>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'FrontC) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'FrontR) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'FrontRR) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'FrontL) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'FrontLL) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'BackC) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'BackR) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'BackL) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <IRs>) istream)
  "Deserializes a message object of type '<IRs>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'FrontC) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'FrontR) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'FrontRR) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'FrontL) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'FrontLL) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'BackC) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'BackR) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'BackL) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<IRs>)))
  "Returns string type for a message object of type '<IRs>"
  "robobo_msgs/IRs")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IRs)))
  "Returns string type for a message object of type 'IRs"
  "robobo_msgs/IRs")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<IRs>)))
  "Returns md5sum for a message object of type '<IRs>"
  "3cda0da1598fe1278d12814909cfb3d7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'IRs)))
  "Returns md5sum for a message object of type 'IRs"
  "3cda0da1598fe1278d12814909cfb3d7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<IRs>)))
  "Returns full string definition for message of type '<IRs>"
  (cl:format cl:nil "sensor_msgs/Range FrontC~%sensor_msgs/Range FrontR~%sensor_msgs/Range FrontRR~%sensor_msgs/Range FrontL~%sensor_msgs/Range FrontLL~%sensor_msgs/Range BackC~%sensor_msgs/Range BackR~%sensor_msgs/Range BackL~%~%================================================================================~%MSG: sensor_msgs/Range~%# Single range reading from an active ranger that emits energy and reports~%# one range reading that is valid along an arc at the distance measured. ~%# This message is  not appropriate for laser scanners. See the LaserScan~%# message if you are working with a laser scanner.~%~%# This message also can represent a fixed-distance (binary) ranger.  This~%# sensor will have min_range===max_range===distance of detection.~%# These sensors follow REP 117 and will output -Inf if the object is detected~%# and +Inf if the object is outside of the detection range.~%~%Header header           # timestamp in the header is the time the ranger~%                        # returned the distance reading~%~%# Radiation type enums~%# If you want a value added to this list, send an email to the ros-users list~%uint8 ULTRASOUND=0~%uint8 INFRARED=1~%~%uint8 radiation_type    # the type of radiation used by the sensor~%                        # (sound, IR, etc) [enum]~%~%float32 field_of_view   # the size of the arc that the distance reading is~%                        # valid for [rad]~%                        # the object causing the range reading may have~%                        # been anywhere within -field_of_view/2 and~%                        # field_of_view/2 at the measured range. ~%                        # 0 angle corresponds to the x-axis of the sensor.~%~%float32 min_range       # minimum range value [m]~%float32 max_range       # maximum range value [m]~%                        # Fixed distance rangers require min_range==max_range~%~%float32 range           # range data [m]~%                        # (Note: values < range_min or > range_max~%                        # should be discarded)~%                        # Fixed distance rangers only output -Inf or +Inf.~%                        # -Inf represents a detection within fixed distance.~%                        # (Detection too close to the sensor to quantify)~%                        # +Inf represents no detection within the fixed distance.~%                        # (Object out of range)~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'IRs)))
  "Returns full string definition for message of type 'IRs"
  (cl:format cl:nil "sensor_msgs/Range FrontC~%sensor_msgs/Range FrontR~%sensor_msgs/Range FrontRR~%sensor_msgs/Range FrontL~%sensor_msgs/Range FrontLL~%sensor_msgs/Range BackC~%sensor_msgs/Range BackR~%sensor_msgs/Range BackL~%~%================================================================================~%MSG: sensor_msgs/Range~%# Single range reading from an active ranger that emits energy and reports~%# one range reading that is valid along an arc at the distance measured. ~%# This message is  not appropriate for laser scanners. See the LaserScan~%# message if you are working with a laser scanner.~%~%# This message also can represent a fixed-distance (binary) ranger.  This~%# sensor will have min_range===max_range===distance of detection.~%# These sensors follow REP 117 and will output -Inf if the object is detected~%# and +Inf if the object is outside of the detection range.~%~%Header header           # timestamp in the header is the time the ranger~%                        # returned the distance reading~%~%# Radiation type enums~%# If you want a value added to this list, send an email to the ros-users list~%uint8 ULTRASOUND=0~%uint8 INFRARED=1~%~%uint8 radiation_type    # the type of radiation used by the sensor~%                        # (sound, IR, etc) [enum]~%~%float32 field_of_view   # the size of the arc that the distance reading is~%                        # valid for [rad]~%                        # the object causing the range reading may have~%                        # been anywhere within -field_of_view/2 and~%                        # field_of_view/2 at the measured range. ~%                        # 0 angle corresponds to the x-axis of the sensor.~%~%float32 min_range       # minimum range value [m]~%float32 max_range       # maximum range value [m]~%                        # Fixed distance rangers require min_range==max_range~%~%float32 range           # range data [m]~%                        # (Note: values < range_min or > range_max~%                        # should be discarded)~%                        # Fixed distance rangers only output -Inf or +Inf.~%                        # -Inf represents a detection within fixed distance.~%                        # (Detection too close to the sensor to quantify)~%                        # +Inf represents no detection within the fixed distance.~%                        # (Object out of range)~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <IRs>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'FrontC))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'FrontR))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'FrontRR))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'FrontL))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'FrontLL))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'BackC))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'BackR))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'BackL))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <IRs>))
  "Converts a ROS message object to a list"
  (cl:list 'IRs
    (cl:cons ':FrontC (FrontC msg))
    (cl:cons ':FrontR (FrontR msg))
    (cl:cons ':FrontRR (FrontRR msg))
    (cl:cons ':FrontL (FrontL msg))
    (cl:cons ':FrontLL (FrontLL msg))
    (cl:cons ':BackC (BackC msg))
    (cl:cons ':BackR (BackR msg))
    (cl:cons ':BackL (BackL msg))
))
