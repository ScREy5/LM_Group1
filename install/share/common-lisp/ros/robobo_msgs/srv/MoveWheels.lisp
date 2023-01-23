; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude MoveWheels-request.msg.html

(cl:defclass <MoveWheels-request> (roslisp-msg-protocol:ros-message)
  ((lspeed
    :reader lspeed
    :initarg :lspeed
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8))
   (rspeed
    :reader rspeed
    :initarg :rspeed
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8))
   (time
    :reader time
    :initarg :time
    :type std_msgs-msg:Int32
    :initform (cl:make-instance 'std_msgs-msg:Int32))
   (unlockid
    :reader unlockid
    :initarg :unlockid
    :type std_msgs-msg:Int16
    :initform (cl:make-instance 'std_msgs-msg:Int16)))
)

(cl:defclass MoveWheels-request (<MoveWheels-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MoveWheels-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MoveWheels-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<MoveWheels-request> is deprecated: use robobo_msgs-srv:MoveWheels-request instead.")))

(cl:ensure-generic-function 'lspeed-val :lambda-list '(m))
(cl:defmethod lspeed-val ((m <MoveWheels-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:lspeed-val is deprecated.  Use robobo_msgs-srv:lspeed instead.")
  (lspeed m))

(cl:ensure-generic-function 'rspeed-val :lambda-list '(m))
(cl:defmethod rspeed-val ((m <MoveWheels-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:rspeed-val is deprecated.  Use robobo_msgs-srv:rspeed instead.")
  (rspeed m))

(cl:ensure-generic-function 'time-val :lambda-list '(m))
(cl:defmethod time-val ((m <MoveWheels-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:time-val is deprecated.  Use robobo_msgs-srv:time instead.")
  (time m))

(cl:ensure-generic-function 'unlockid-val :lambda-list '(m))
(cl:defmethod unlockid-val ((m <MoveWheels-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:unlockid-val is deprecated.  Use robobo_msgs-srv:unlockid instead.")
  (unlockid m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MoveWheels-request>) ostream)
  "Serializes a message object of type '<MoveWheels-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'lspeed) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rspeed) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'time) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'unlockid) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MoveWheels-request>) istream)
  "Deserializes a message object of type '<MoveWheels-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'lspeed) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rspeed) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'time) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'unlockid) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MoveWheels-request>)))
  "Returns string type for a service object of type '<MoveWheels-request>"
  "robobo_msgs/MoveWheelsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MoveWheels-request)))
  "Returns string type for a service object of type 'MoveWheels-request"
  "robobo_msgs/MoveWheelsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MoveWheels-request>)))
  "Returns md5sum for a message object of type '<MoveWheels-request>"
  "f7d9b6ab6c37c283e7f3f620a3f5c819")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MoveWheels-request)))
  "Returns md5sum for a message object of type 'MoveWheels-request"
  "f7d9b6ab6c37c283e7f3f620a3f5c819")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MoveWheels-request>)))
  "Returns full string definition for message of type '<MoveWheels-request>"
  (cl:format cl:nil "std_msgs/Int8 lspeed~%std_msgs/Int8 rspeed~%std_msgs/Int32 time~%std_msgs/Int16 unlockid~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%================================================================================~%MSG: std_msgs/Int16~%int16 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MoveWheels-request)))
  "Returns full string definition for message of type 'MoveWheels-request"
  (cl:format cl:nil "std_msgs/Int8 lspeed~%std_msgs/Int8 rspeed~%std_msgs/Int32 time~%std_msgs/Int16 unlockid~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%================================================================================~%MSG: std_msgs/Int32~%int32 data~%================================================================================~%MSG: std_msgs/Int16~%int16 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MoveWheels-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'lspeed))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rspeed))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'time))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'unlockid))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MoveWheels-request>))
  "Converts a ROS message object to a list"
  (cl:list 'MoveWheels-request
    (cl:cons ':lspeed (lspeed msg))
    (cl:cons ':rspeed (rspeed msg))
    (cl:cons ':time (time msg))
    (cl:cons ':unlockid (unlockid msg))
))
;//! \htmlinclude MoveWheels-response.msg.html

(cl:defclass <MoveWheels-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass MoveWheels-response (<MoveWheels-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MoveWheels-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MoveWheels-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<MoveWheels-response> is deprecated: use robobo_msgs-srv:MoveWheels-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <MoveWheels-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MoveWheels-response>) ostream)
  "Serializes a message object of type '<MoveWheels-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MoveWheels-response>) istream)
  "Deserializes a message object of type '<MoveWheels-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MoveWheels-response>)))
  "Returns string type for a service object of type '<MoveWheels-response>"
  "robobo_msgs/MoveWheelsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MoveWheels-response)))
  "Returns string type for a service object of type 'MoveWheels-response"
  "robobo_msgs/MoveWheelsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MoveWheels-response>)))
  "Returns md5sum for a message object of type '<MoveWheels-response>"
  "f7d9b6ab6c37c283e7f3f620a3f5c819")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MoveWheels-response)))
  "Returns md5sum for a message object of type 'MoveWheels-response"
  "f7d9b6ab6c37c283e7f3f620a3f5c819")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MoveWheels-response>)))
  "Returns full string definition for message of type '<MoveWheels-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MoveWheels-response)))
  "Returns full string definition for message of type 'MoveWheels-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MoveWheels-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MoveWheels-response>))
  "Converts a ROS message object to a list"
  (cl:list 'MoveWheels-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'MoveWheels)))
  'MoveWheels-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'MoveWheels)))
  'MoveWheels-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MoveWheels)))
  "Returns string type for a service object of type '<MoveWheels>"
  "robobo_msgs/MoveWheels")