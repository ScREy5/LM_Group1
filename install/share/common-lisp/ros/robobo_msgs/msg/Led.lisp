; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-msg)


;//! \htmlinclude Led.msg.html

(cl:defclass <Led> (roslisp-msg-protocol:ros-message)
  ((id
    :reader id
    :initarg :id
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String))
   (color
    :reader color
    :initarg :color
    :type std_msgs-msg:ColorRGBA
    :initform (cl:make-instance 'std_msgs-msg:ColorRGBA)))
)

(cl:defclass Led (<Led>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Led>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Led)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-msg:<Led> is deprecated: use robobo_msgs-msg:Led instead.")))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <Led>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:id-val is deprecated.  Use robobo_msgs-msg:id instead.")
  (id m))

(cl:ensure-generic-function 'color-val :lambda-list '(m))
(cl:defmethod color-val ((m <Led>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-msg:color-val is deprecated.  Use robobo_msgs-msg:color instead.")
  (color m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Led>) ostream)
  "Serializes a message object of type '<Led>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'id) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'color) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Led>) istream)
  "Deserializes a message object of type '<Led>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'id) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'color) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Led>)))
  "Returns string type for a message object of type '<Led>"
  "robobo_msgs/Led")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Led)))
  "Returns string type for a message object of type 'Led"
  "robobo_msgs/Led")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Led>)))
  "Returns md5sum for a message object of type '<Led>"
  "5da60702833e9003bbb8cd608e95a6bd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Led)))
  "Returns md5sum for a message object of type 'Led"
  "5da60702833e9003bbb8cd608e95a6bd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Led>)))
  "Returns full string definition for message of type '<Led>"
  (cl:format cl:nil "std_msgs/String id~%std_msgs/ColorRGBA color~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Led)))
  "Returns full string definition for message of type 'Led"
  (cl:format cl:nil "std_msgs/String id~%std_msgs/ColorRGBA color~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Led>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'id))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'color))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Led>))
  "Converts a ROS message object to a list"
  (cl:list 'Led
    (cl:cons ':id (id msg))
    (cl:cons ':color (color msg))
))
