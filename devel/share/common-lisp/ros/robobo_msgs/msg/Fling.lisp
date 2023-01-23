; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-msg)


;//! \htmlinclude Fling.msg.html

(cl:defclass <Fling> (roslisp-msg-protocol:ros-message)
  ((angle
    :reader angle
    :initarg :angle
    :type std_msgs-msg:Int16
    :initform (cl:make-instance 'std_msgs-msg:Int16))
   (time
    :reader time
    :initarg :time
    :type std_msgs-msg:Int32
    :initform (cl:make-instance 'std_msgs-msg:Int32))
   (distance
    :reader distance
    :initarg :distance
    :type std_msgs-msg:Int16
    :initform (cl:make-instance 'std_msgs-msg:Int16)))
)

(cl:defclass Fling (<Fling>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Fling>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Fling)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-msg:<Fling> is deprecated: use robobo_msgs-msg:Fling instead.")))

(cl:ensure-generic-function 'angle-val :lambda-list '(m))
(cl:defmethod angle-val ((m <Fling>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:angle-val is deprecated.  Use robobo_msgs-msg:angle instead.")
  (angle m))

(cl:ensure-generic-function 'time-val :lambda-list '(m))
(cl:defmethod time-val ((m <Fling>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:time-val is deprecated.  Use robobo_msgs-msg:time instead.")
  (time m))

(cl:ensure-generic-function 'distance-val :lambda-list '(m))
(cl:defmethod distance-val ((m <Fling>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:distance-val is deprecated.  Use robobo_msgs-msg:distance instead.")
  (distance m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Fling>) ostream)
  "Serializes a message object of type '<Fling>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'angle) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'time) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'distance) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Fling>) istream)
  "Deserializes a message object of type '<Fling>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'angle) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'time) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'distance) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Fling>)))
  "Returns string type for a message object of type '<Fling>"
  "robobo_msgs/Fling")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Fling)))
  "Returns string type for a message object of type 'Fling"
  "robobo_msgs/Fling")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Fling>)))
  "Returns md5sum for a message object of type '<Fling>"
  "223606e2db34b4f54252a5c13a50f185")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Fling)))
  "Returns md5sum for a message object of type 'Fling"
  "223606e2db34b4f54252a5c13a50f185")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Fling>)))
  "Returns full string definition for message of type '<Fling>"
  (cl:format cl:nil "std_msgs/Int16 angle~%std_msgs/Int32 time~%std_msgs/Int16 distance~%~%================================================================================~%MSG: std_msgs/Int16~%int16 data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Fling)))
  "Returns full string definition for message of type 'Fling"
  (cl:format cl:nil "std_msgs/Int16 angle~%std_msgs/Int32 time~%std_msgs/Int16 distance~%~%================================================================================~%MSG: std_msgs/Int16~%int16 data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Fling>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'angle))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'time))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'distance))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Fling>))
  "Converts a ROS message object to a list"
  (cl:list 'Fling
    (cl:cons ':angle (angle msg))
    (cl:cons ':time (time msg))
    (cl:cons ':distance (distance msg))
))
