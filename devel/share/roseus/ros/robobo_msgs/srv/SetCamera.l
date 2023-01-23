;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::SetCamera)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'SetCamera (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::SETCAMERA")
  (make-package "ROBOBO_MSGS::SETCAMERA"))
(unless (find-package "ROBOBO_MSGS::SETCAMERAREQUEST")
  (make-package "ROBOBO_MSGS::SETCAMERAREQUEST"))
(unless (find-package "ROBOBO_MSGS::SETCAMERARESPONSE")
  (make-package "ROBOBO_MSGS::SETCAMERARESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::SetCameraRequest
  :super ros::object
  :slots (_camera ))

(defmethod robobo_msgs::SetCameraRequest
  (:init
   (&key
    ((:camera __camera) (instance std_msgs::Int8 :init))
    )
   (send-super :init)
   (setq _camera __camera)
   self)
  (:camera
   (&rest __camera)
   (if (keywordp (car __camera))
       (send* _camera __camera)
     (progn
       (if __camera (setq _camera (car __camera)))
       _camera)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8 _camera
    (send _camera :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8 _camera
       (send _camera :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8 _camera
     (send _camera :deserialize buf ptr-) (incf ptr- (send _camera :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::SetCameraResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::SetCameraResponse
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

(defclass robobo_msgs::SetCamera
  :super ros::object
  :slots ())

(setf (get robobo_msgs::SetCamera :md5sum-) "90a7c1d0dd1fb7ca39545c447b7dd045")
(setf (get robobo_msgs::SetCamera :datatype-) "robobo_msgs/SetCamera")
(setf (get robobo_msgs::SetCamera :request) robobo_msgs::SetCameraRequest)
(setf (get robobo_msgs::SetCamera :response) robobo_msgs::SetCameraResponse)

(defmethod robobo_msgs::SetCameraRequest
  (:response () (instance robobo_msgs::SetCameraResponse :init)))

(setf (get robobo_msgs::SetCameraRequest :md5sum-) "90a7c1d0dd1fb7ca39545c447b7dd045")
(setf (get robobo_msgs::SetCameraRequest :datatype-) "robobo_msgs/SetCameraRequest")
(setf (get robobo_msgs::SetCameraRequest :definition-)
      "std_msgs/Int8 camera   #0 for front camera
                       #1 for back cameras

================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/Int8 error 


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::SetCameraResponse :md5sum-) "90a7c1d0dd1fb7ca39545c447b7dd045")
(setf (get robobo_msgs::SetCameraResponse :datatype-) "robobo_msgs/SetCameraResponse")
(setf (get robobo_msgs::SetCameraResponse :definition-)
      "std_msgs/Int8 camera   #0 for front camera
                       #1 for back cameras

================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/Int8 error 


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/SetCamera "90a7c1d0dd1fb7ca39545c447b7dd045")


