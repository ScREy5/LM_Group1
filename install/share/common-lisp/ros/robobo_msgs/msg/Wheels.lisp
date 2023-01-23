; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-msg)


;//! \htmlinclude Wheels.msg.html

(cl:defclass <Wheels> (roslisp-msg-protocol:ros-message)
  ((wheelPosR
    :reader wheelPosR
    :initarg :wheelPosR
    :type std_msgs-msg:Int64
    :initform (cl:make-instance 'std_msgs-msg:Int64))
   (wheelPosL
    :reader wheelPosL
    :initarg :wheelPosL
    :type std_msgs-msg:Int64
    :initform (cl:make-instance 'std_msgs-msg:Int64))
   (wheelSpeedR
    :reader wheelSpeedR
    :initarg :wheelSpeedR
    :type std_msgs-msg:Int64
    :initform (cl:make-instance 'std_msgs-msg:Int64))
   (wheelSpeedL
    :reader wheelSpeedL
    :initarg :wheelSpeedL
    :type std_msgs-msg:Int64
    :initform (cl:make-instance 'std_msgs-msg:Int64)))
)

(cl:defclass Wheels (<Wheels>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Wheels>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Wheels)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-msg:<Wheels> is deprecated: use robobo_msgs-msg:Wheels instead.")))

(cl:ensure-generic-function 'wheelPosR-val :lambda-list '(m))
(cl:defmethod wheelPosR-val ((m <Wheels>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:wheelPosR-val is deprecated.  Use robobo_msgs-msg:wheelPosR instead.")
  (wheelPosR m))

(cl:ensure-generic-function 'wheelPosL-val :lambda-list '(m))
(cl:defmethod wheelPosL-val ((m <Wheels>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:wheelPosL-val is deprecated.  Use robobo_msgs-msg:wheelPosL instead.")
  (wheelPosL m))

(cl:ensure-generic-function 'wheelSpeedR-val :lambda-list '(m))
(cl:defmethod wheelSpeedR-val ((m <Wheels>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:wheelSpeedR-val is deprecated.  Use robobo_msgs-msg:wheelSpeedR instead.")
  (wheelSpeedR m))

(cl:ensure-generic-function 'wheelSpeedL-val :lambda-list '(m))
(cl:defmethod wheelSpeedL-val ((m <Wheels>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:wheelSpeedL-val is deprecated.  Use robobo_msgs-msg:wheelSpeedL instead.")
  (wheelSpeedL m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Wheels>) ostream)
  "Serializes a message object of type '<Wheels>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'wheelPosR) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'wheelPosL) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'wheelSpeedR) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'wheelSpeedL) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Wheels>) istream)
  "Deserializes a message object of type '<Wheels>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'wheelPosR) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'wheelPosL) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'wheelSpeedR) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'wheelSpeedL) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Wheels>)))
  "Returns string type for a message object of type '<Wheels>"
  "robobo_msgs/Wheels")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Wheels)))
  "Returns string type for a message object of type 'Wheels"
  "robobo_msgs/Wheels")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Wheels>)))
  "Returns md5sum for a message object of type '<Wheels>"
  "e3007c5c2ec7faa3f8f71bd3a0d670c5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Wheels)))
  "Returns md5sum for a message object of type 'Wheels"
  "e3007c5c2ec7faa3f8f71bd3a0d670c5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Wheels>)))
  "Returns full string definition for message of type '<Wheels>"
  (cl:format cl:nil "std_msgs/Int64 wheelPosR~%std_msgs/Int64 wheelPosL~%std_msgs/Int64 wheelSpeedR~%std_msgs/Int64 wheelSpeedL~%~%================================================================================~%MSG: std_msgs/Int64~%int64 data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Wheels)))
  "Returns full string definition for message of type 'Wheels"
  (cl:format cl:nil "std_msgs/Int64 wheelPosR~%std_msgs/Int64 wheelPosL~%std_msgs/Int64 wheelSpeedR~%std_msgs/Int64 wheelSpeedL~%~%================================================================================~%MSG: std_msgs/Int64~%int64 data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Wheels>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'wheelPosR))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'wheelPosL))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'wheelSpeedR))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'wheelSpeedL))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Wheels>))
  "Converts a ROS message object to a list"
  (cl:list 'Wheels
    (cl:cons ':wheelPosR (wheelPosR msg))
    (cl:cons ':wheelPosL (wheelPosL msg))
    (cl:cons ':wheelSpeedR (wheelSpeedR msg))
    (cl:cons ':wheelSpeedL (wheelSpeedL msg))
))
