; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude SetLed-request.msg.html

(cl:defclass <SetLed-request> (roslisp-msg-protocol:ros-message)
  ((id
    :reader id
    :initarg :id
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String))
   (color
    :reader color
    :initarg :color
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass SetLed-request (<SetLed-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetLed-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetLed-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetLed-request> is deprecated: use robobo_msgs-srv:SetLed-request instead.")))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <SetLed-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:id-val is deprecated.  Use robobo_msgs-srv:id instead.")
  (id m))

(cl:ensure-generic-function 'color-val :lambda-list '(m))
(cl:defmethod color-val ((m <SetLed-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:color-val is deprecated.  Use robobo_msgs-srv:color instead.")
  (color m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetLed-request>) ostream)
  "Serializes a message object of type '<SetLed-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'id) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'color) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetLed-request>) istream)
  "Deserializes a message object of type '<SetLed-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'id) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'color) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetLed-request>)))
  "Returns string type for a service object of type '<SetLed-request>"
  "robobo_msgs/SetLedRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetLed-request)))
  "Returns string type for a service object of type 'SetLed-request"
  "robobo_msgs/SetLedRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetLed-request>)))
  "Returns md5sum for a message object of type '<SetLed-request>"
  "cae286d48c8a0ad77943140106a4ad56")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetLed-request)))
  "Returns md5sum for a message object of type 'SetLed-request"
  "cae286d48c8a0ad77943140106a4ad56")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetLed-request>)))
  "Returns full string definition for message of type '<SetLed-request>"
  (cl:format cl:nil "std_msgs/String id~%std_msgs/String color~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetLed-request)))
  "Returns full string definition for message of type 'SetLed-request"
  (cl:format cl:nil "std_msgs/String id~%std_msgs/String color~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetLed-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'id))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'color))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetLed-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetLed-request
    (cl:cons ':id (id msg))
    (cl:cons ':color (color msg))
))
;//! \htmlinclude SetLed-response.msg.html

(cl:defclass <SetLed-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass SetLed-response (<SetLed-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetLed-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetLed-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetLed-response> is deprecated: use robobo_msgs-srv:SetLed-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <SetLed-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetLed-response>) ostream)
  "Serializes a message object of type '<SetLed-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetLed-response>) istream)
  "Deserializes a message object of type '<SetLed-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetLed-response>)))
  "Returns string type for a service object of type '<SetLed-response>"
  "robobo_msgs/SetLedResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetLed-response)))
  "Returns string type for a service object of type 'SetLed-response"
  "robobo_msgs/SetLedResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetLed-response>)))
  "Returns md5sum for a message object of type '<SetLed-response>"
  "cae286d48c8a0ad77943140106a4ad56")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetLed-response)))
  "Returns md5sum for a message object of type 'SetLed-response"
  "cae286d48c8a0ad77943140106a4ad56")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetLed-response>)))
  "Returns full string definition for message of type '<SetLed-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetLed-response)))
  "Returns full string definition for message of type 'SetLed-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetLed-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetLed-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetLed-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetLed)))
  'SetLed-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetLed)))
  'SetLed-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetLed)))
  "Returns string type for a service object of type '<SetLed>"
  "robobo_msgs/SetLed")