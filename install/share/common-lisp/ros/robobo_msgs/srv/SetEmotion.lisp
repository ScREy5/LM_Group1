; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude SetEmotion-request.msg.html

(cl:defclass <SetEmotion-request> (roslisp-msg-protocol:ros-message)
  ((emotion
    :reader emotion
    :initarg :emotion
    :type std_msgs-msg:String
    :initform (cl:make-instance 'std_msgs-msg:String)))
)

(cl:defclass SetEmotion-request (<SetEmotion-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetEmotion-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetEmotion-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetEmotion-request> is deprecated: use robobo_msgs-srv:SetEmotion-request instead.")))

(cl:ensure-generic-function 'emotion-val :lambda-list '(m))
(cl:defmethod emotion-val ((m <SetEmotion-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:emotion-val is deprecated.  Use robobo_msgs-srv:emotion instead.")
  (emotion m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetEmotion-request>) ostream)
  "Serializes a message object of type '<SetEmotion-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'emotion) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetEmotion-request>) istream)
  "Deserializes a message object of type '<SetEmotion-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'emotion) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetEmotion-request>)))
  "Returns string type for a service object of type '<SetEmotion-request>"
  "robobo_msgs/SetEmotionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetEmotion-request)))
  "Returns string type for a service object of type 'SetEmotion-request"
  "robobo_msgs/SetEmotionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetEmotion-request>)))
  "Returns md5sum for a message object of type '<SetEmotion-request>"
  "b56a22e94c3e6cd89d04a799670be723")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetEmotion-request)))
  "Returns md5sum for a message object of type 'SetEmotion-request"
  "b56a22e94c3e6cd89d04a799670be723")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetEmotion-request>)))
  "Returns full string definition for message of type '<SetEmotion-request>"
  (cl:format cl:nil "std_msgs/String emotion~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetEmotion-request)))
  "Returns full string definition for message of type 'SetEmotion-request"
  (cl:format cl:nil "std_msgs/String emotion~%~%================================================================================~%MSG: std_msgs/String~%string data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetEmotion-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'emotion))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetEmotion-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetEmotion-request
    (cl:cons ':emotion (emotion msg))
))
;//! \htmlinclude SetEmotion-response.msg.html

(cl:defclass <SetEmotion-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass SetEmotion-response (<SetEmotion-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetEmotion-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetEmotion-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetEmotion-response> is deprecated: use robobo_msgs-srv:SetEmotion-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <SetEmotion-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetEmotion-response>) ostream)
  "Serializes a message object of type '<SetEmotion-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetEmotion-response>) istream)
  "Deserializes a message object of type '<SetEmotion-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetEmotion-response>)))
  "Returns string type for a service object of type '<SetEmotion-response>"
  "robobo_msgs/SetEmotionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetEmotion-response)))
  "Returns string type for a service object of type 'SetEmotion-response"
  "robobo_msgs/SetEmotionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetEmotion-response>)))
  "Returns md5sum for a message object of type '<SetEmotion-response>"
  "b56a22e94c3e6cd89d04a799670be723")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetEmotion-response)))
  "Returns md5sum for a message object of type 'SetEmotion-response"
  "b56a22e94c3e6cd89d04a799670be723")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetEmotion-response>)))
  "Returns full string definition for message of type '<SetEmotion-response>"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetEmotion-response)))
  "Returns full string definition for message of type 'SetEmotion-response"
  (cl:format cl:nil "std_msgs/Int8 error~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetEmotion-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetEmotion-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetEmotion-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetEmotion)))
  'SetEmotion-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetEmotion)))
  'SetEmotion-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetEmotion)))
  "Returns string type for a service object of type '<SetEmotion>"
  "robobo_msgs/SetEmotion")