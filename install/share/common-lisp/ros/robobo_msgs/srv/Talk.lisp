; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude Talk-request.msg.html

(cl:defclass <Talk-request> (roslisp-msg-protocol:ros-message)
  ((text
    :reader text
    :initarg :text
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass Talk-request (<Talk-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Talk-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Talk-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<Talk-request> is deprecated: use robobo_msgs-srv:Talk-request instead.")))

(cl:ensure-generic-function 'text-val :lambda-list '(m))
(cl:defmethod text-val ((m <Talk-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:text-val is deprecated.  Use robobo_msgs-srv:text instead.")
  (text m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Talk-request>) ostream)
  "Serializes a message object of type '<Talk-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'text) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Talk-request>) istream)
  "Deserializes a message object of type '<Talk-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'text) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Talk-request>)))
  "Returns string type for a service object of type '<Talk-request>"
  "robobo_msgs/TalkRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Talk-request)))
  "Returns string type for a service object of type 'Talk-request"
  "robobo_msgs/TalkRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Talk-request>)))
  "Returns md5sum for a message object of type '<Talk-request>"
  "d8fdbf7e3675ec38cfc5f64bfa512f70")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Talk-request)))
  "Returns md5sum for a message object of type 'Talk-request"
  "d8fdbf7e3675ec38cfc5f64bfa512f70")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Talk-request>)))
  "Returns full string definition for message of type '<Talk-request>"
  (cl:format cl:nil "std_msgs/String text~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Talk-request)))
  "Returns full string definition for message of type 'Talk-request"
  (cl:format cl:nil "std_msgs/String text~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Talk-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'text))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Talk-request>))
  "Converts a ROS message object to a list"
  (cl:list 'Talk-request
    (cl:cons ':text (text msg))
))
;//! \htmlinclude Talk-response.msg.html

(cl:defclass <Talk-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass Talk-response (<Talk-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Talk-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Talk-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<Talk-response> is deprecated: use robobo_msgs-srv:Talk-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <Talk-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Talk-response>) ostream)
  "Serializes a message object of type '<Talk-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Talk-response>) istream)
  "Deserializes a message object of type '<Talk-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Talk-response>)))
  "Returns string type for a service object of type '<Talk-response>"
  "robobo_msgs/TalkResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Talk-response)))
  "Returns string type for a service object of type 'Talk-response"
  "robobo_msgs/TalkResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Talk-response>)))
  "Returns md5sum for a message object of type '<Talk-response>"
  "d8fdbf7e3675ec38cfc5f64bfa512f70")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Talk-response)))
  "Returns md5sum for a message object of type 'Talk-response"
  "d8fdbf7e3675ec38cfc5f64bfa512f70")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Talk-response>)))
  "Returns full string definition for message of type '<Talk-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Talk-response)))
  "Returns full string definition for message of type 'Talk-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Talk-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Talk-response>))
  "Converts a ROS message object to a list"
  (cl:list 'Talk-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'Talk)))
  'Talk-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'Talk)))
  'Talk-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Talk)))
  "Returns string type for a service object of type '<Talk>"
  "robobo_msgs/Talk")