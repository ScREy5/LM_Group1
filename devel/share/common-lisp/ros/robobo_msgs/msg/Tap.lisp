; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-msg)


;//! \htmlinclude Tap.msg.html

(cl:defclass <Tap> (roslisp-msg-protocol:ros-message)
  ((x
    :reader x
    :initarg :x
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8))
   (y
    :reader y
    :initarg :y
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass Tap (<Tap>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Tap>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Tap)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-msg:<Tap> is deprecated: use robobo_msgs-msg:Tap instead.")))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <Tap>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:x-val is deprecated.  Use robobo_msgs-msg:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <Tap>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:y-val is deprecated.  Use robobo_msgs-msg:y instead.")
  (y m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Tap>) ostream)
  "Serializes a message object of type '<Tap>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'x) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'y) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Tap>) istream)
  "Deserializes a message object of type '<Tap>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'x) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'y) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Tap>)))
  "Returns string type for a message object of type '<Tap>"
  "robobo_msgs/Tap")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Tap)))
  "Returns string type for a message object of type 'Tap"
  "robobo_msgs/Tap")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Tap>)))
  "Returns md5sum for a message object of type '<Tap>"
  "c455a05a0b8c9dc9827ed72d23a4de17")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Tap)))
  "Returns md5sum for a message object of type 'Tap"
  "c455a05a0b8c9dc9827ed72d23a4de17")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Tap>)))
  "Returns full string definition for message of type '<Tap>"
  (cl:format cl:nil "std_msgs/Int8 x~%std_msgs/Int8 y~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Tap)))
  "Returns full string definition for message of type 'Tap"
  (cl:format cl:nil "std_msgs/Int8 x~%std_msgs/Int8 y~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Tap>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'x))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'y))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Tap>))
  "Converts a ROS message object to a list"
  (cl:list 'Tap
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
))
