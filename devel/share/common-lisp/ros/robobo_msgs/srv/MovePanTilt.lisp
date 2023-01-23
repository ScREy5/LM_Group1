; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude MovePanTilt-request.msg.html

(cl:defclass <MovePanTilt-request> (roslisp-msg-protocol:ros-message)
  ((panPos
    :reader panPos
    :initarg :panPos
    :type std_msgs-msg:Int16
    :initform (cl:make-instance 'std_msgs-msg:Int16))
   (panSpeed
    :reader panSpeed
    :initarg :panSpeed
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8))
   (panUnlockId
    :reader panUnlockId
    :initarg :panUnlockId
    :type std_msgs-msg:Int16
    :initform (cl:make-instance 'std_msgs-msg:Int16))
   (tiltPos
    :reader tiltPos
    :initarg :tiltPos
    :type std_msgs-msg:Int16
    :initform (cl:make-instance 'std_msgs-msg:Int16))
   (tiltSpeed
    :reader tiltSpeed
    :initarg :tiltSpeed
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8))
   (tiltUnlockId
    :reader tiltUnlockId
    :initarg :tiltUnlockId
    :type std_msgs-msg:Int16
    :initform (cl:make-instance 'std_msgs-msg:Int16)))
)

(cl:defclass MovePanTilt-request (<MovePanTilt-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MovePanTilt-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MovePanTilt-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<MovePanTilt-request> is deprecated: use robobo_msgs-srv:MovePanTilt-request instead.")))

(cl:ensure-generic-function 'panPos-val :lambda-list '(m))
(cl:defmethod panPos-val ((m <MovePanTilt-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:panPos-val is deprecated.  Use robobo_msgs-srv:panPos instead.")
  (panPos m))

(cl:ensure-generic-function 'panSpeed-val :lambda-list '(m))
(cl:defmethod panSpeed-val ((m <MovePanTilt-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:panSpeed-val is deprecated.  Use robobo_msgs-srv:panSpeed instead.")
  (panSpeed m))

(cl:ensure-generic-function 'panUnlockId-val :lambda-list '(m))
(cl:defmethod panUnlockId-val ((m <MovePanTilt-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:panUnlockId-val is deprecated.  Use robobo_msgs-srv:panUnlockId instead.")
  (panUnlockId m))

(cl:ensure-generic-function 'tiltPos-val :lambda-list '(m))
(cl:defmethod tiltPos-val ((m <MovePanTilt-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:tiltPos-val is deprecated.  Use robobo_msgs-srv:tiltPos instead.")
  (tiltPos m))

(cl:ensure-generic-function 'tiltSpeed-val :lambda-list '(m))
(cl:defmethod tiltSpeed-val ((m <MovePanTilt-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:tiltSpeed-val is deprecated.  Use robobo_msgs-srv:tiltSpeed instead.")
  (tiltSpeed m))

(cl:ensure-generic-function 'tiltUnlockId-val :lambda-list '(m))
(cl:defmethod tiltUnlockId-val ((m <MovePanTilt-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:tiltUnlockId-val is deprecated.  Use robobo_msgs-srv:tiltUnlockId instead.")
  (tiltUnlockId m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MovePanTilt-request>) ostream)
  "Serializes a message object of type '<MovePanTilt-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'panPos) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'panSpeed) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'panUnlockId) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tiltPos) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tiltSpeed) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tiltUnlockId) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MovePanTilt-request>) istream)
  "Deserializes a message object of type '<MovePanTilt-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'panPos) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'panSpeed) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'panUnlockId) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tiltPos) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tiltSpeed) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tiltUnlockId) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MovePanTilt-request>)))
  "Returns string type for a service object of type '<MovePanTilt-request>"
  "robobo_msgs/MovePanTiltRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MovePanTilt-request)))
  "Returns string type for a service object of type 'MovePanTilt-request"
  "robobo_msgs/MovePanTiltRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MovePanTilt-request>)))
  "Returns md5sum for a message object of type '<MovePanTilt-request>"
  "cf92955e4ca6d280225977f6e00d71c8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MovePanTilt-request)))
  "Returns md5sum for a message object of type 'MovePanTilt-request"
  "cf92955e4ca6d280225977f6e00d71c8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MovePanTilt-request>)))
  "Returns full string definition for message of type '<MovePanTilt-request>"
  (cl:format cl:nil "std_msgs/Int16 panPos~%std_msgs/Int8 panSpeed~%std_msgs/Int16 panUnlockId~%std_msgs/Int16 tiltPos~%std_msgs/Int8 tiltSpeed~%std_msgs/Int16 tiltUnlockId~%~%================================================================================~%MSG: std_msgs/Int16~%int16 data~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MovePanTilt-request)))
  "Returns full string definition for message of type 'MovePanTilt-request"
  (cl:format cl:nil "std_msgs/Int16 panPos~%std_msgs/Int8 panSpeed~%std_msgs/Int16 panUnlockId~%std_msgs/Int16 tiltPos~%std_msgs/Int8 tiltSpeed~%std_msgs/Int16 tiltUnlockId~%~%================================================================================~%MSG: std_msgs/Int16~%int16 data~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MovePanTilt-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'panPos))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'panSpeed))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'panUnlockId))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tiltPos))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tiltSpeed))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tiltUnlockId))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MovePanTilt-request>))
  "Converts a ROS message object to a list"
  (cl:list 'MovePanTilt-request
    (cl:cons ':panPos (panPos msg))
    (cl:cons ':panSpeed (panSpeed msg))
    (cl:cons ':panUnlockId (panUnlockId msg))
    (cl:cons ':tiltPos (tiltPos msg))
    (cl:cons ':tiltSpeed (tiltSpeed msg))
    (cl:cons ':tiltUnlockId (tiltUnlockId msg))
))
;//! \htmlinclude MovePanTilt-response.msg.html

(cl:defclass <MovePanTilt-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass MovePanTilt-response (<MovePanTilt-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MovePanTilt-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MovePanTilt-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<MovePanTilt-response> is deprecated: use robobo_msgs-srv:MovePanTilt-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <MovePanTilt-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MovePanTilt-response>) ostream)
  "Serializes a message object of type '<MovePanTilt-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MovePanTilt-response>) istream)
  "Deserializes a message object of type '<MovePanTilt-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MovePanTilt-response>)))
  "Returns string type for a service object of type '<MovePanTilt-response>"
  "robobo_msgs/MovePanTiltResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MovePanTilt-response)))
  "Returns string type for a service object of type 'MovePanTilt-response"
  "robobo_msgs/MovePanTiltResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MovePanTilt-response>)))
  "Returns md5sum for a message object of type '<MovePanTilt-response>"
  "cf92955e4ca6d280225977f6e00d71c8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MovePanTilt-response)))
  "Returns md5sum for a message object of type 'MovePanTilt-response"
  "cf92955e4ca6d280225977f6e00d71c8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MovePanTilt-response>)))
  "Returns full string definition for message of type '<MovePanTilt-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MovePanTilt-response)))
  "Returns full string definition for message of type 'MovePanTilt-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MovePanTilt-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MovePanTilt-response>))
  "Converts a ROS message object to a list"
  (cl:list 'MovePanTilt-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'MovePanTilt)))
  'MovePanTilt-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'MovePanTilt)))
  'MovePanTilt-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MovePanTilt)))
  "Returns string type for a service object of type '<MovePanTilt>"
  "robobo_msgs/MovePanTilt")