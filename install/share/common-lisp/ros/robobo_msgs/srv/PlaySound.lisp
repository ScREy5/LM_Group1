; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude PlaySound-request.msg.html

(cl:defclass <PlaySound-request> (roslisp-msg-protocol:ros-message)
  ((sound
    :reader sound
    :initarg :sound
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass PlaySound-request (<PlaySound-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PlaySound-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PlaySound-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<PlaySound-request> is deprecated: use robobo_msgs-srv:PlaySound-request instead.")))

(cl:ensure-generic-function 'sound-val :lambda-list '(m))
(cl:defmethod sound-val ((m <PlaySound-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:sound-val is deprecated.  Use robobo_msgs-srv:sound instead.")
  (sound m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PlaySound-request>) ostream)
  "Serializes a message object of type '<PlaySound-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'sound) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PlaySound-request>) istream)
  "Deserializes a message object of type '<PlaySound-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'sound) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PlaySound-request>)))
  "Returns string type for a service object of type '<PlaySound-request>"
  "robobo_msgs/PlaySoundRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PlaySound-request)))
  "Returns string type for a service object of type 'PlaySound-request"
  "robobo_msgs/PlaySoundRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PlaySound-request>)))
  "Returns md5sum for a message object of type '<PlaySound-request>"
  "c95eacdcbf61c615623911c64d3b591f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PlaySound-request)))
  "Returns md5sum for a message object of type 'PlaySound-request"
  "c95eacdcbf61c615623911c64d3b591f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PlaySound-request>)))
  "Returns full string definition for message of type '<PlaySound-request>"
  (cl:format cl:nil "std_msgs/String sound~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PlaySound-request)))
  "Returns full string definition for message of type 'PlaySound-request"
  (cl:format cl:nil "std_msgs/String sound~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PlaySound-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'sound))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PlaySound-request>))
  "Converts a ROS message object to a list"
  (cl:list 'PlaySound-request
    (cl:cons ':sound (sound msg))
))
;//! \htmlinclude PlaySound-response.msg.html

(cl:defclass <PlaySound-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass PlaySound-response (<PlaySound-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PlaySound-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PlaySound-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<PlaySound-response> is deprecated: use robobo_msgs-srv:PlaySound-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <PlaySound-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PlaySound-response>) ostream)
  "Serializes a message object of type '<PlaySound-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PlaySound-response>) istream)
  "Deserializes a message object of type '<PlaySound-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PlaySound-response>)))
  "Returns string type for a service object of type '<PlaySound-response>"
  "robobo_msgs/PlaySoundResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PlaySound-response)))
  "Returns string type for a service object of type 'PlaySound-response"
  "robobo_msgs/PlaySoundResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PlaySound-response>)))
  "Returns md5sum for a message object of type '<PlaySound-response>"
  "c95eacdcbf61c615623911c64d3b591f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PlaySound-response)))
  "Returns md5sum for a message object of type 'PlaySound-response"
  "c95eacdcbf61c615623911c64d3b591f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PlaySound-response>)))
  "Returns full string definition for message of type '<PlaySound-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PlaySound-response)))
  "Returns full string definition for message of type 'PlaySound-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PlaySound-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PlaySound-response>))
  "Converts a ROS message object to a list"
  (cl:list 'PlaySound-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'PlaySound)))
  'PlaySound-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'PlaySound)))
  'PlaySound-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PlaySound)))
  "Returns string type for a service object of type '<PlaySound>"
  "robobo_msgs/PlaySound")