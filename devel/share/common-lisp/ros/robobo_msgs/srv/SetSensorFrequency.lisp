; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude SetSensorFrequency-request.msg.html

(cl:defclass <SetSensorFrequency-request> (roslisp-msg-protocol:ros-message)
  ((frequency
    :reader frequency
    :initarg :frequency
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass SetSensorFrequency-request (<SetSensorFrequency-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSensorFrequency-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSensorFrequency-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetSensorFrequency-request> is deprecated: use robobo_msgs-srv:SetSensorFrequency-request instead.")))

(cl:ensure-generic-function 'frequency-val :lambda-list '(m))
(cl:defmethod frequency-val ((m <SetSensorFrequency-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:frequency-val is deprecated.  Use robobo_msgs-srv:frequency instead.")
  (frequency m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSensorFrequency-request>) ostream)
  "Serializes a message object of type '<SetSensorFrequency-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'frequency) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSensorFrequency-request>) istream)
  "Deserializes a message object of type '<SetSensorFrequency-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'frequency) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSensorFrequency-request>)))
  "Returns string type for a service object of type '<SetSensorFrequency-request>"
  "robobo_msgs/SetSensorFrequencyRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSensorFrequency-request)))
  "Returns string type for a service object of type 'SetSensorFrequency-request"
  "robobo_msgs/SetSensorFrequencyRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSensorFrequency-request>)))
  "Returns md5sum for a message object of type '<SetSensorFrequency-request>"
  "ea3a75b016df1ca9629d8a0e272b0401")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSensorFrequency-request)))
  "Returns md5sum for a message object of type 'SetSensorFrequency-request"
  "ea3a75b016df1ca9629d8a0e272b0401")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSensorFrequency-request>)))
  "Returns full string definition for message of type '<SetSensorFrequency-request>"
  (cl:format cl:nil "std_msgs/Int8 frequency   #0 for LOW~%                          #1 for NORMAL~%                          #2 for HIGH~%                          #3 for MAX~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSensorFrequency-request)))
  "Returns full string definition for message of type 'SetSensorFrequency-request"
  (cl:format cl:nil "std_msgs/Int8 frequency   #0 for LOW~%                          #1 for NORMAL~%                          #2 for HIGH~%                          #3 for MAX~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSensorFrequency-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'frequency))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSensorFrequency-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSensorFrequency-request
    (cl:cons ':frequency (frequency msg))
))
;//! \htmlinclude SetSensorFrequency-response.msg.html

(cl:defclass <SetSensorFrequency-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass SetSensorFrequency-response (<SetSensorFrequency-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSensorFrequency-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSensorFrequency-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetSensorFrequency-response> is deprecated: use robobo_msgs-srv:SetSensorFrequency-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <SetSensorFrequency-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSensorFrequency-response>) ostream)
  "Serializes a message object of type '<SetSensorFrequency-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSensorFrequency-response>) istream)
  "Deserializes a message object of type '<SetSensorFrequency-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSensorFrequency-response>)))
  "Returns string type for a service object of type '<SetSensorFrequency-response>"
  "robobo_msgs/SetSensorFrequencyResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSensorFrequency-response)))
  "Returns string type for a service object of type 'SetSensorFrequency-response"
  "robobo_msgs/SetSensorFrequencyResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSensorFrequency-response>)))
  "Returns md5sum for a message object of type '<SetSensorFrequency-response>"
  "ea3a75b016df1ca9629d8a0e272b0401")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSensorFrequency-response)))
  "Returns md5sum for a message object of type 'SetSensorFrequency-response"
  "ea3a75b016df1ca9629d8a0e272b0401")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSensorFrequency-response>)))
  "Returns full string definition for message of type '<SetSensorFrequency-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSensorFrequency-response)))
  "Returns full string definition for message of type 'SetSensorFrequency-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSensorFrequency-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSensorFrequency-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSensorFrequency-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetSensorFrequency)))
  'SetSensorFrequency-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetSensorFrequency)))
  'SetSensorFrequency-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSensorFrequency)))
  "Returns string type for a service object of type '<SetSensorFrequency>"
  "robobo_msgs/SetSensorFrequency")