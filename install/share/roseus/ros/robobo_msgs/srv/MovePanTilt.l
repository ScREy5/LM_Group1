;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::MovePanTilt)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'MovePanTilt (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::MOVEPANTILT")
  (make-package "ROBOBO_MSGS::MOVEPANTILT"))
(unless (find-package "ROBOBO_MSGS::MOVEPANTILTREQUEST")
  (make-package "ROBOBO_MSGS::MOVEPANTILTREQUEST"))
(unless (find-package "ROBOBO_MSGS::MOVEPANTILTRESPONSE")
  (make-package "ROBOBO_MSGS::MOVEPANTILTRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::MovePanTiltRequest
  :super ros::object
  :slots (_panPos _panSpeed _panUnlockId _tiltPos _tiltSpeed _tiltUnlockId ))

(defmethod robobo_msgs::MovePanTiltRequest
  (:init
   (&key
    ((:panPos __panPos) (instance std_msgs::Int16 :init))
    ((:panSpeed __panSpeed) (instance std_msgs::Int8 :init))
    ((:panUnlockId __panUnlockId) (instance std_msgs::Int16 :init))
    ((:tiltPos __tiltPos) (instance std_msgs::Int16 :init))
    ((:tiltSpeed __tiltSpeed) (instance std_msgs::Int8 :init))
    ((:tiltUnlockId __tiltUnlockId) (instance std_msgs::Int16 :init))
    )
   (send-super :init)
   (setq _panPos __panPos)
   (setq _panSpeed __panSpeed)
   (setq _panUnlockId __panUnlockId)
   (setq _tiltPos __tiltPos)
   (setq _tiltSpeed __tiltSpeed)
   (setq _tiltUnlockId __tiltUnlockId)
   self)
  (:panPos
   (&rest __panPos)
   (if (keywordp (car __panPos))
       (send* _panPos __panPos)
     (progn
       (if __panPos (setq _panPos (car __panPos)))
       _panPos)))
  (:panSpeed
   (&rest __panSpeed)
   (if (keywordp (car __panSpeed))
       (send* _panSpeed __panSpeed)
     (progn
       (if __panSpeed (setq _panSpeed (car __panSpeed)))
       _panSpeed)))
  (:panUnlockId
   (&rest __panUnlockId)
   (if (keywordp (car __panUnlockId))
       (send* _panUnlockId __panUnlockId)
     (progn
       (if __panUnlockId (setq _panUnlockId (car __panUnlockId)))
       _panUnlockId)))
  (:tiltPos
   (&rest __tiltPos)
   (if (keywordp (car __tiltPos))
       (send* _tiltPos __tiltPos)
     (progn
       (if __tiltPos (setq _tiltPos (car __tiltPos)))
       _tiltPos)))
  (:tiltSpeed
   (&rest __tiltSpeed)
   (if (keywordp (car __tiltSpeed))
       (send* _tiltSpeed __tiltSpeed)
     (progn
       (if __tiltSpeed (setq _tiltSpeed (car __tiltSpeed)))
       _tiltSpeed)))
  (:tiltUnlockId
   (&rest __tiltUnlockId)
   (if (keywordp (car __tiltUnlockId))
       (send* _tiltUnlockId __tiltUnlockId)
     (progn
       (if __tiltUnlockId (setq _tiltUnlockId (car __tiltUnlockId)))
       _tiltUnlockId)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int16 _panPos
    (send _panPos :serialization-length)
    ;; std_msgs/Int8 _panSpeed
    (send _panSpeed :serialization-length)
    ;; std_msgs/Int16 _panUnlockId
    (send _panUnlockId :serialization-length)
    ;; std_msgs/Int16 _tiltPos
    (send _tiltPos :serialization-length)
    ;; std_msgs/Int8 _tiltSpeed
    (send _tiltSpeed :serialization-length)
    ;; std_msgs/Int16 _tiltUnlockId
    (send _tiltUnlockId :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int16 _panPos
       (send _panPos :serialize s)
     ;; std_msgs/Int8 _panSpeed
       (send _panSpeed :serialize s)
     ;; std_msgs/Int16 _panUnlockId
       (send _panUnlockId :serialize s)
     ;; std_msgs/Int16 _tiltPos
       (send _tiltPos :serialize s)
     ;; std_msgs/Int8 _tiltSpeed
       (send _tiltSpeed :serialize s)
     ;; std_msgs/Int16 _tiltUnlockId
       (send _tiltUnlockId :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int16 _panPos
     (send _panPos :deserialize buf ptr-) (incf ptr- (send _panPos :serialization-length))
   ;; std_msgs/Int8 _panSpeed
     (send _panSpeed :deserialize buf ptr-) (incf ptr- (send _panSpeed :serialization-length))
   ;; std_msgs/Int16 _panUnlockId
     (send _panUnlockId :deserialize buf ptr-) (incf ptr- (send _panUnlockId :serialization-length))
   ;; std_msgs/Int16 _tiltPos
     (send _tiltPos :deserialize buf ptr-) (incf ptr- (send _tiltPos :serialization-length))
   ;; std_msgs/Int8 _tiltSpeed
     (send _tiltSpeed :deserialize buf ptr-) (incf ptr- (send _tiltSpeed :serialization-length))
   ;; std_msgs/Int16 _tiltUnlockId
     (send _tiltUnlockId :deserialize buf ptr-) (incf ptr- (send _tiltUnlockId :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::MovePanTiltResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::MovePanTiltResponse
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

(defclass robobo_msgs::MovePanTilt
  :super ros::object
  :slots ())

(setf (get robobo_msgs::MovePanTilt :md5sum-) "cf92955e4ca6d280225977f6e00d71c8")
(setf (get robobo_msgs::MovePanTilt :datatype-) "robobo_msgs/MovePanTilt")
(setf (get robobo_msgs::MovePanTilt :request) robobo_msgs::MovePanTiltRequest)
(setf (get robobo_msgs::MovePanTilt :response) robobo_msgs::MovePanTiltResponse)

(defmethod robobo_msgs::MovePanTiltRequest
  (:response () (instance robobo_msgs::MovePanTiltResponse :init)))

(setf (get robobo_msgs::MovePanTiltRequest :md5sum-) "cf92955e4ca6d280225977f6e00d71c8")
(setf (get robobo_msgs::MovePanTiltRequest :datatype-) "robobo_msgs/MovePanTiltRequest")
(setf (get robobo_msgs::MovePanTiltRequest :definition-)
      "std_msgs/Int16 panPos
std_msgs/Int8 panSpeed
std_msgs/Int16 panUnlockId
std_msgs/Int16 tiltPos
std_msgs/Int8 tiltSpeed
std_msgs/Int16 tiltUnlockId

================================================================================
MSG: std_msgs/Int16
int16 data

================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::MovePanTiltResponse :md5sum-) "cf92955e4ca6d280225977f6e00d71c8")
(setf (get robobo_msgs::MovePanTiltResponse :datatype-) "robobo_msgs/MovePanTiltResponse")
(setf (get robobo_msgs::MovePanTiltResponse :definition-)
      "std_msgs/Int16 panPos
std_msgs/Int8 panSpeed
std_msgs/Int16 panUnlockId
std_msgs/Int16 tiltPos
std_msgs/Int8 tiltSpeed
std_msgs/Int16 tiltUnlockId

================================================================================
MSG: std_msgs/Int16
int16 data

================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/MovePanTilt "cf92955e4ca6d280225977f6e00d71c8")


