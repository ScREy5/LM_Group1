; Auto-generated. Do not edit!


(cl:in-package robobo_msgs-srv)


;//! \htmlinclude SetCamera-request.msg.html

(cl:defclass <SetCamera-request> (roslisp-msg-protocol:ros-message)
  ((camera
    :reader camera
    :initarg :camera
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass SetCamera-request (<SetCamera-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetCamera-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetCamera-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetCamera-request> is deprecated: use robobo_msgs-srv:SetCamera-request instead.")))

(cl:ensure-generic-function 'camera-val :lambda-list '(m))
(cl:defmethod camera-val ((m <SetCamera-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:camera-val is deprecated.  Use robobo_msgs-srv:camera instead.")
  (camera m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetCamera-request>) ostream)
  "Serializes a message object of type '<SetCamera-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'camera) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetCamera-request>) istream)
  "Deserializes a message object of type '<SetCamera-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'camera) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetCamera-request>)))
  "Returns string type for a service object of type '<SetCamera-request>"
  "robobo_msgs/SetCameraRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCamera-request)))
  "Returns string type for a service object of type 'SetCamera-request"
  "robobo_msgs/SetCameraRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetCamera-request>)))
  "Returns md5sum for a message object of type '<SetCamera-request>"
  "90a7c1d0dd1fb7ca39545c447b7dd045")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetCamera-request)))
  "Returns md5sum for a message object of type 'SetCamera-request"
  "90a7c1d0dd1fb7ca39545c447b7dd045")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetCamera-request>)))
  "Returns full string definition for message of type '<SetCamera-request>"
  (cl:format cl:nil "std_msgs/Int8 camera   #0 for front camera~%                       #1 for back cameras~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetCamera-request)))
  "Returns full string definition for message of type 'SetCamera-request"
  (cl:format cl:nil "std_msgs/Int8 camera   #0 for front camera~%                       #1 for back cameras~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetCamera-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'camera))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetCamera-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetCamera-request
    (cl:cons ':camera (camera msg))
))
;//! \htmlinclude SetCamera-response.msg.html

(cl:defclass <SetCamera-response> (roslisp-msg-protocol:ros-message)
  ((error
    :reader error
    :initarg :error
    :type std_msgs-msg:Int8
    :initform (cl:make-instance 'std_msgs-msg:Int8)))
)

(cl:defclass SetCamera-response (<SetCamera-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetCamera-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetCamera-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name robobo_msgs-srv:<SetCamera-response> is deprecated: use robobo_msgs-srv:SetCamera-response instead.")))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <SetCamera-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader robobo_msgs-srv:error-val is deprecated.  Use robobo_msgs-srv:error instead.")
  (error m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetCamera-response>) ostream)
  "Serializes a message object of type '<SetCamera-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'error) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetCamera-response>) istream)
  "Deserializes a message object of type '<SetCamera-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'error) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetCamera-response>)))
  "Returns string type for a service object of type '<SetCamera-response>"
  "robobo_msgs/SetCameraResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCamera-response)))
  "Returns string type for a service object of type 'SetCamera-response"
  "robobo_msgs/SetCameraResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetCamera-response>)))
  "Returns md5sum for a message object of type '<SetCamera-response>"
  "90a7c1d0dd1fb7ca39545c447b7dd045")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetCamera-response)))
  "Returns md5sum for a message object of type 'SetCamera-response"
  "90a7c1d0dd1fb7ca39545c447b7dd045")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetCamera-response>)))
  "Returns full string definition for message of type '<SetCamera-response>"
  (cl:format cl:nil "std_msgs/Int8 error ~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetCamera-response)))
  "Returns full string definition for message of type 'SetCamera-response"
  (cl:format cl:nil "std_msgs/Int8 error ~%~%~%================================================================================~%MSG: std_msgs/Int8~%int8 data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetCamera-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'error))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetCamera-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetCamera-response
    (cl:cons ':error (error msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetCamera)))
  'SetCamera-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetCamera)))
  'SetCamera-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetCamera)))
  "Returns string type for a service object of type '<SetCamera>"
  "robobo_msgs/SetCamera")