;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::ResetWheels)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'ResetWheels (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::RESETWHEELS")
  (make-package "ROBOBO_MSGS::RESETWHEELS"))
(unless (find-package "ROBOBO_MSGS::RESETWHEELSREQUEST")
  (make-package "ROBOBO_MSGS::RESETWHEELSREQUEST"))
(unless (find-package "ROBOBO_MSGS::RESETWHEELSRESPONSE")
  (make-package "ROBOBO_MSGS::RESETWHEELSRESPONSE"))

(in-package "ROS")



(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::ResetWheelsRequest
  :super ros::object
  :slots ())

(defmethod robobo_msgs::ResetWheelsRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass robobo_msgs::ResetWheelsResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::ResetWheelsResponse
  (:init
   (&key
    ((:error __error) (instance std_msgs::Int8 :init))
    )
   (send-super :init)
   (setq _error __error)
   self)
  (:error
   (&rest __error)
   (if (keywordp (car __error))
       (send* _error __error)
     (progn
       (if __error (setq _error (car __error)))
       _error)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8 _error
    (send _error :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8 _error
       (send _error :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8 _error
     (send _error :deserialize buf ptr-) (incf ptr- (send _error :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::ResetWheels
  :super ros::object
  :slots ())

(setf (get robobo_msgs::ResetWheels :md5sum-) "b41202f44ec8802b6a66ae15859258a4")
(setf (get robobo_msgs::ResetWheels :datatype-) "robobo_msgs/ResetWheels")
(setf (get robobo_msgs::ResetWheels :request) robobo_msgs::ResetWheelsRequest)
(setf (get robobo_msgs::ResetWheels :response) robobo_msgs::ResetWheelsResponse)

(defmethod robobo_msgs::ResetWheelsRequest
  (:response () (instance robobo_msgs::ResetWheelsResponse :init)))

(setf (get robobo_msgs::ResetWheelsRequest :md5sum-) "b41202f44ec8802b6a66ae15859258a4")
(setf (get robobo_msgs::ResetWheelsRequest :datatype-) "robobo_msgs/ResetWheelsRequest")
(setf (get robobo_msgs::ResetWheelsRequest :definition-)
      "---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::ResetWheelsResponse :md5sum-) "b41202f44ec8802b6a66ae15859258a4")
(setf (get robobo_msgs::ResetWheelsResponse :datatype-) "robobo_msgs/ResetWheelsResponse")
(setf (get robobo_msgs::ResetWheelsResponse :definition-)
      "---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/ResetWheels "b41202f44ec8802b6a66ae15859258a4")


