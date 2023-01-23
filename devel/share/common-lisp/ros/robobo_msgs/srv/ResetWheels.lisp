; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude ResetWheels-request.msg.html

(cl:defclass <ResetWheels-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass ResetWheels-request (<ResetWheels-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResetWheels-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResetWheels-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<ResetWheels-request> is deprecated: use robobo_msgs-srv:ResetWheels-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResetWheels-request>) ostream)
  "Serializes a message object of type '<ResetWheels-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResetWheels-request>) istream)
  "Deserializes a message object of type '<ResetWheels-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResetWheels-request>)))
  "Returns string type for a service object of type '<ResetWheels-request>"
  "robobo_msgs/ResetWheelsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetWheels-request)))
  "Returns string type for a service object of type 'ResetWheels-request"
  "robobo_msgs/ResetWheelsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResetWheels-request>)))
  "Returns md5sum for a message object of type '<ResetWheels-request>"
  "b41202f44ec8802b6a66ae15859258a4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResetWheels-request)))
  "Returns md5sum for a message object of type 'ResetWheels-request"
  "b41202f44ec8802b6a66ae15859258a4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResetWheels-request>)))
  "Returns full string definition for message of type '<ResetWheels-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResetWheels-request)))
  "Returns full string definition for message of type 'ResetWheels-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResetWheels-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResetWheels-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ResetWheels-request
))
;//! \htmlinclude ResetWheels-response.msg.html

(cl:defclass <ResetWheels-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass ResetWheels-response (<ResetWheels-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResetWheels-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResetWheels-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<ResetWheels-response> is deprecated: use robobo_msgs-srv:ResetWheels-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <ResetWheels-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResetWheels-response>) ostream)
  "Serializes a message object of type '<ResetWheels-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResetWheels-response>) istream)
  "Deserializes a message object of type '<ResetWheels-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResetWheels-response>)))
  "Returns string type for a service object of type '<ResetWheels-response>"
  "robobo_msgs/ResetWheelsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetWheels-response)))
  "Returns string type for a service object of type 'ResetWheels-response"
  "robobo_msgs/ResetWheelsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResetWheels-response>)))
  "Returns md5sum for a message object of type '<ResetWheels-response>"
  "b41202f44ec8802b6a66ae15859258a4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResetWheels-response)))
  "Returns md5sum for a message object of type 'ResetWheels-response"
  "b41202f44ec8802b6a66ae15859258a4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResetWheels-response>)))
  "Returns full string definition for message of type '<ResetWheels-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResetWheels-response)))
  "Returns full string definition for message of type 'ResetWheels-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResetWheels-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResetWheels-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ResetWheels-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ResetWheels)))
  'ResetWheels-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ResetWheels)))
  'ResetWheels-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetWheels)))
  "Returns string type for a service object of type '<ResetWheels>"
  "robobo_msgs/ResetWheels")